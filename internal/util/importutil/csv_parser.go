// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package importutil

import (
	"context"
	"encoding/csv"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strings"

	"github.com/milvus-io/milvus/internal/storage"
	"github.com/milvus-io/milvus/pkg/log"
	"github.com/milvus-io/milvus/pkg/util/typeutil"
	"go.uber.org/zap"
)

type CSVParser struct {
	ctx	context.Context
	collectionInfo *CollectionInfo
	bufRowCount int
	fieldsName []string
	updateProgressFunc func(percent int64)
}

func NewCSVParser(ctx context.Context, collectionInfo *CollectionInfo, updateProgressFunc func(percent int64)) (*CSVParser, error) {
	if collectionInfo == nil {
		log.Warn("CSV parser: collection schema is nil")
		return nil, errors.New("collection schema is nil")
	}

	parser := &CSVParser{
		ctx: ctx,
		collectionInfo: collectionInfo,
		bufRowCount: 1024,
		fieldsName: make([]string, 0),
		updateProgressFunc: updateProgressFunc,
	}
	parser.SetBufSize()
	return parser, nil
}

func (p *CSVParser) SetBufSize() {
	schema := p.collectionInfo.Schema
	sizePerRecord, _ := typeutil.EstimateSizePerRecord(schema)
	if sizePerRecord <= 0 {
		return
	}

	bufRowCount := p.bufRowCount
	for {
		if bufRowCount * sizePerRecord > SingleBlockSize {
			bufRowCount--
		} else {
			break
		}
	}
	if bufRowCount <= 0 {
		bufRowCount = 1
	}
	log.Info("CSV parser: reset bufRowCount", zap.Int("sizePerRecord", sizePerRecord), zap.Int("bufRowCount", bufRowCount))
	p.bufRowCount = bufRowCount
}

func (p *CSVParser) combineDynamicRow(dynamicValues map[string]string, row map[storage.FieldID]string) error {
	if p.collectionInfo.DynamicField == nil {
		return nil
	}

	dynamicFieldID := p.collectionInfo.DynamicField.GetFieldID()
	// combine the dynamic field value
	// valid input:
	//        id,vector,x,$meta        id,vector,$meta
	// case1: 1,"[]",8,"{""y"": 8}" ==>> 1,"[]","{""y"": 8, ""x"": 8}"
	// case2: 1,"[]",8,"{}"       ==>> 1,"[]","{""x"": 8}"
	// case3: 1,"[]",,"{""x"": 8}"
	// case4: 1,"[]",8,            ==>> 1,"[]","{""x"": 8}"
	// case5: 1,"[]",,
	value, ok := row[dynamicFieldID]

	if len(dynamicValues) > 0 {
		mp := make(map[string]interface{})
		if ok {
			// case 1/2
			// $meta is JSON type field, we first convert it to map[string]interface{}
			// then merge other dynamic field into it
			desc := json.NewDecoder(strings.NewReader(value))
			desc.UseNumber()
			if err := desc.Decode(&mp); err != nil {
				log.Warn("CSV parser: illegal value for dynamic field, not a JSON object")
				return errors.New("illegal value for dynamic field, not a JSON object")
			}
		}
		// case 4
		// 已经给出的动态字段没有类型，如果直接转化为json会导致类型错误（全部都为 string ），所以在这里需要额外判断
		for k, v := range dynamicValues {
			desc := json.NewDecoder(strings.NewReader(v))
			desc.UseNumber()
			var v2json interface{}
			err := desc.Decode(&v2json)
			if err != nil {
				// 视为字符串
				mp[k] = v
			} else {
				mp[k] = v2json
			}
		}
		bs, err := json.Marshal(mp)
		if err != nil {
			log.Warn("CSV parser: illegal value for dynamic field, not a JSON object")
			return errors.New("illegal value for dynamic field, not a JSON object")
		}
		row[dynamicFieldID] = string(bs)
	} else if !ok && len(dynamicValues) == 0 {
		// case 5
		row[dynamicFieldID] = "{}"
	}
	// else case 3

	return nil
}


func (p *CSVParser) verifyRow(raw []string) (map[storage.FieldID]string, error) {
	row := make(map[storage.FieldID]string)
	dynamicValues := make(map[string]string)
	// the size of raw and fieldsName must be same
	if len(raw) != len(p.fieldsName) {
		log.Warn("CSV parser: some field value is missed", zap.Int("inputSize", len(raw)), zap.Int("fieldSize", len(p.fieldsName)))
		return nil, errors.New("some field value is missed")
	}

	for i := 0; i < len(p.fieldsName); i++ {
		fieldName := p.fieldsName[i]
		fieldID, ok := p.collectionInfo.Name2FieldID[fieldName]

		if fieldID == p.collectionInfo.PrimaryKey.GetFieldID() && p.collectionInfo.PrimaryKey.GetAutoID() {
			// primary key is auto-id, no need to provide
			log.Warn("CSV parser: the primary key is auto-generated, no need to provide", zap.String("fieldName", fieldName))
			return nil, fmt.Errorf("the primary key '%s' is auto-generated, no need to provide", fieldName)
		}

		if ok {
			row[fieldID] = raw[i]
		} else if p.collectionInfo.DynamicField != nil {
			// collection have dynamic field. put it to dynamicValues
			dynamicValues[fieldName] = raw[i]
		} else {
			// no dynamic field. if user provided redundant field, return error
			log.Warn("CSV parser: the field is not defined in collection schema", zap.String("fieldName", fieldName))
			return nil, fmt.Errorf("the field '%s' is not defined in collection schema", fieldName)
		}
	}
	// some fields not provided?
	if len(row) != len(p.collectionInfo.Name2FieldID) {
		for k, v := range p.collectionInfo.Name2FieldID {
			if p.collectionInfo.DynamicField != nil && v == p.collectionInfo.DynamicField.GetFieldID() {
				// ignore dyanmic field, user don't have to provide values for dynamic field
				continue
			}

			if v == p.collectionInfo.PrimaryKey.GetFieldID() && p.collectionInfo.PrimaryKey.GetAutoID() {
				// ignore auto-generaed primary key
				continue
			}
			_, ok := row[v]
			if !ok {
				// not auto-id primary key, no dynamic field,  must provide value
				log.Warn("CSV parser: a field value is missed", zap.String("fieldName", k))
				return nil, fmt.Errorf("value of field '%s' is missed", k)
			}

		}
	}
	// combine the redundant pairs into dynamic field(if has)
	err := p.combineDynamicRow(dynamicValues, row)
	if err != nil {
		log.Warn("CSV parser: failed to combine dynamic values", zap.Error(err))
		return nil, err
	}

	return row, nil
}

func (p *CSVParser) ParseRows(reader *IOReader, handle CSVRowHandler) error {
	if reader == nil || handle == nil {
		log.Warn("CSV Parser: CSV parse handle is nil")
		return errors.New("CSV parse handle is nil")
	}
	r := csv.NewReader(reader.r)
	
	oldPercent := int64(0)
	updateProgress := func() {
		if p.updateProgressFunc != nil && reader.fileSize > 0 {
			percent := (r.InputOffset() * ProgressValueForPersist) / reader.fileSize
			if percent > oldPercent { // avoid too many log
				log.Debug("CSV parser: working progress", zap.Int64("offset", r.InputOffset()),
					zap.Int64("fileSize", reader.fileSize), zap.Int64("percent", percent))
			}
			oldPercent = percent
			p.updateProgressFunc(percent)
		}
	}

	isEmpty := true
	for {

		// read the fields value
		fieldsName, err := r.Read()
		if err == io.EOF {
			break
		} else if err != nil {
			log.Warn("CSV Parser: failed to parse the field value", zap.Error(err))
			return fmt.Errorf("failed to read the field value, error: %w", err)
		}
		log.Info("解析字段信息成功", zap.Strings("fieldsName", fieldsName))
		p.fieldsName = fieldsName
		// read buffer
		buf := make([]map[storage.FieldID]string, 0, p.bufRowCount)
		for {
			// read the row value
			values, err := r.Read()
			
			if err == io.EOF {
				log.Info("CSV parser: work done")
				break
			} else if err != nil {
				log.Warn("CSV parser: failed to parse row value", zap.Error(err))
				return fmt.Errorf("failed to parse row value, error: %w", err)
			}

			row, err := p.verifyRow(values)
			if err != nil {
				return err
			}
			
			updateProgress()

			buf = append(buf, row)
			if len(buf) >= p.bufRowCount {
				isEmpty = false
				log.Info("送入消费者消费", zap.Int("bufferSize", len(buf)))
				if err = handle.Handle(buf); err != nil {
					log.Warn("CSV parser: failed to convert row value to entity", zap.Error(err))
					return fmt.Errorf("failed to convert row value to entity, error: %w", err)
				}
				// clean the buffer
				buf = make([]map[storage.FieldID]string, 0, p.bufRowCount)
			}
		}
		if len(buf) > 0 {
			isEmpty = false
			if err = handle.Handle(buf); err != nil {
				log.Warn("CSV parser: failed to convert row value to entity", zap.Error(err))
				return fmt.Errorf("failed to convert row value to entity, error: %w", err)
			}
		}

		// outside context might be canceled(service stop, or future enhancement for canceling import task)
		if isCanceled(p.ctx)  {
			log.Warn("CSV parser: import task was canceled")
			return errors.New("import task was canceled")
		}

		break
	}
	
	// empty file is allowed, don't return error
	if isEmpty {
		log.Info("CSV Parser: row count is 0")
		return nil
	}

	updateProgress()

	return handle.Handle(nil)
}