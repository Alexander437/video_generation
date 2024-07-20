import {PictureOutlined, PlusOutlined} from '@ant-design/icons';
import {Image, Upload, message, Collapse, Button} from 'antd';
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


function UploadImage (props: {
    fileList: UploadFile[],
    onChange: UploadProps['onChange'],
    onPreview: UploadProps['onPreview'],
    generateImage: any,
    selectedImage: File | null,
}) {

  const uploadButton = (
    <button style={{ border: 0, background: 'none' }} type="button">
      <PlusOutlined />
      <div style={{ marginTop: 8 }}>Загрузка</div>
    </button>
  );

  return (
      <div>
          <Collapse
              defaultActiveKey={['1']}
              size="small"
              items={[{
                  key: '1', label: 'Выбрать изображение',
                  children:
                      <Upload
                          action="http://localhost:8000/video/upload"
                          listType="picture-card"
                          fileList={props.fileList}
                          onPreview={props.onPreview}
                          onChange={props.onChange}
                          beforeUpload={beforeUpload}
                      >
                          {props.fileList.length >= 8 ? null : uploadButton}
                      </Upload>
              }]} />
          <Collapse
              size="small"
              items={[{
                  key: '1', label: 'Сгенерировать изображение',
                  children:
                      <div className="flex flex-col items-center my-4">
                          {/*<Select*/}
                          {/*    mode="multiple"*/}
                          {/*    style={{ width: '100%' }}*/}
                          {/*    placeholder="Описание аватара"*/}
                          {/*    onChange={onGenImgChange}*/}
                          {/*    options={genImgOptions}*/}
                          {/*/>*/}
                          <Button
                              type="primary"
                              onClick={props.generateImage}
                          >
                              Сгенерировать
                          </Button>
                      </div>
              }]}/>
          <div className="flex flex-col items-center mt-4">
              <h3 className="text-blue-950 text-xl mb-2">Изображение:</h3>
              {props.selectedImage ? (
                  <Image
                      src={URL.createObjectURL(props.selectedImage)}
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
}

export default UploadImage;
