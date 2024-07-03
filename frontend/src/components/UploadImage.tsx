import React from 'react';
import {PictureOutlined, PlusOutlined} from '@ant-design/icons';
import {Image, Upload, message} from 'antd';
import type { GetProp, UploadFile, UploadProps } from 'antd';


type FileType = Parameters<GetProp<UploadProps, 'beforeUpload'>>[0];

const beforeUpload = (file: FileType) => {
  const isJpgOrPng = file.type === 'image/jpeg' || file.type === 'image/png';
  if (!isJpgOrPng) {
    message.error('You can only upload JPG/PNG file!');
  }
  const isLt2M = file.size / 1024 / 1024 < 2;
  if (!isLt2M) {
    message.error('Image must smaller than 2MB!');
  }
  return isJpgOrPng && isLt2M;
};

const UploadImage: React.FC<{
  fileList: UploadFile[];
  onChange: UploadProps['onChange'];
  onPreview: UploadProps['onPreview'];
  selectedImage: File | null;
}> = ({ fileList, onChange, onPreview, selectedImage }) => {


  const uploadButton = (
    <button style={{ border: 0, background: 'none' }} type="button">
      <PlusOutlined />
      <div style={{ marginTop: 8 }}>Загрузка</div>
    </button>
  );

  return (
      <div>
          <Upload
              action="http://localhost:8000/video/upload"
              listType="picture-card"
              fileList={fileList}
              onPreview={onPreview}
              onChange={onChange}
              beforeUpload={beforeUpload}
          >
              {fileList.length >= 8 ? null : uploadButton}
          </Upload>
          <div className="flex flex-col items-center mt-4">
              <h3 className="text-blue-950 text-xl mb-2">Выбранное изображение:</h3>
              {selectedImage ? (
                  <Image
                      src={URL.createObjectURL(selectedImage)}
                      className="max-h-1/5"
                      style={{maxHeight: '20vh'}}
                  />
              ) : (
                  <div className="flex flex-col items-center justify-center max-h-1/5" style={{maxHeight: '20vh'}}>
                      <PictureOutlined style={{fontSize: '4rem', color: '#ccc', height: '20vh'}}/>
                  </div>
              )}
          </div>
      </div>
  );
};

export default UploadImage;
