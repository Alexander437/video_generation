import React, {useState} from "react";
import {generateVideo} from "./api/generateVideo.ts";
import {Button, Card, Input, Row, Space, Spin, type UploadFile, type UploadProps} from "antd";
import UploadImage from "./components/UploadImage.tsx";
import {LoadingOutlined, YoutubeOutlined} from "@ant-design/icons";


function App() {
    const [videoUrl, setVideoUrl] = useState<string | null>(null);
    const [fileList, setFileList] = useState<UploadFile[]>([]);
    const [selectedImage, setSelectedImage] = useState<File | null>(null);
    const [inputText, setInputText] = useState<string>("Привет");
    const [loading, setLoading] = useState<boolean>(false);


    const handlePreview = async (file: UploadFile) => {
       const fileObj = file.originFileObj || (await fetch(file.url).then(r => r.blob()));
        setSelectedImage(fileObj as File);
  };

  const handleImageChange: UploadProps['onChange'] = ({ fileList: newFileList }) => {
      if (newFileList.length > 3) {
          newFileList = newFileList.slice(1);
      }
    setFileList(newFileList);
    if (newFileList.length > 0) {
      const file = newFileList.at(-1).originFileObj;
      setSelectedImage(file);
    } else {
      setSelectedImage(null);
    }
  }

  const handleInputChange = (text: any) => {
      setInputText(text.target.value);
  }

    const handleGenerateVideo = async () => {
        if (selectedImage) {
            const speaker = "aidar";
            const sampleRate = 8000;

            try {
                setLoading(true);
                const videoBlob = await generateVideo({
                    imgFile: selectedImage,
                    text: inputText,
                    speaker: speaker,
                    sampleRate: sampleRate,
                });
                const videoUrl = URL.createObjectURL(videoBlob);
                setVideoUrl(videoUrl);
                setLoading(false);
            }
            catch (error) {
                console.error(error);
            }
        }
    }

  return (
      <>
          <div className="relative">
              <img src="dls.jpg" alt="Background" className="w-full h-auto"/>
              <h1 className="absolute inset-0 flex items-center justify-center text-yellow-500 text-5xl">
                  Генерация видеоаватара
              </h1>
          </div>
          <Row justify="space-around">
              {/* Левая часть экрана */}
              <Card style={{width: '40%'}}>
                  <UploadImage
                      fileList={fileList}
                      onChange={handleImageChange}
                      onPreview={handlePreview}
                      selectedImage={selectedImage}
                  />
                  <div className="flex flex-col items-center my-4">
                      <Space.Compact style={{ width: '100%' }}>
                          <Input defaultValue={inputText} onChange={handleInputChange} />
                          <Button type="primary" onClick={handleGenerateVideo}>Сгенерировать видео</Button>
                      </Space.Compact>
                  </div>
              </Card>
              {/* Правая часть экрана */}
              <Card style={{width: '40%'}}>
                  <h3 className="text-blue-950 text-xl">Видео:</h3>
                  <div className="flex flex-col items-center justify-center max-h-1/5" style={{height: '41vh'}}>
                      {loading ? (
                          <Spin indicator={<LoadingOutlined style={{fontSize: 52}} spin/>}/>
                      ) : (
                          <>
                              {videoUrl ? (
                                  <div className="h-96">
                                      <video controls className="h-full rounded-lg">
                                          <source src={videoUrl} type="video/mp4"/>
                                      </video>
                                  </div>
                              ) : (
                                  <YoutubeOutlined style={{fontSize: '4rem', color: '#ccc', height: '41vh'}}/>
                              )}
                          </>
                      )}
                  </div>
              </Card>
          </Row>
      </>
)
}

export default App
