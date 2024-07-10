import React, {useState} from "react";
import {generateVideo} from "./api/generateVideo.ts";
import {Button, Card, Input, Row, Select, Space, Spin, type UploadFile, type UploadProps} from "antd";
import UploadImage from "./components/UploadImage.tsx";
import {LoadingOutlined, YoutubeOutlined} from "@ant-design/icons";

const { TextArea } = Input;

function App() {
    const [videoUrl, setVideoUrl] = useState<string | null>(null);
    const [fileList, setFileList] = useState<UploadFile[]>([]);
    const [loading, setLoading] = useState<boolean>(false);

    const [selectedImage, setSelectedImage] = useState<File | null>(null);
    const [inputText, setInputText] = useState<string>("Привет");
    const [speaker, setSpeaker] = useState<string>("aidar");
    const [sampleRate, setSampleRate] = useState<number>(24000);


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

  const handlePreview = async (file: UploadFile) => {
      const fileObj = file.originFileObj || (await fetch(file.url).then(r => r.blob()));
      setSelectedImage(fileObj as File);
  };

  const handleSpeakerSelect = (value: string) => setSpeaker(value)
  const handleSampleRateSelect = (value: number) => setSampleRate(value)
  const handleInputChange = (text: any) => setInputText(text.target.value)

    const handleGenerateVideo = async () => {
        if (selectedImage) {
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
                      <div className="flex flex-col items-center text-blue-950 text-lg">Текст:</div>
                      <TextArea defaultValue={inputText} onChange={handleInputChange} style={{width: '1000'}}/>
                      <Row className="my-2">
                          <Space>
                              <div className="text-blue-950 text-lg">Голос:</div>
                              <Select
                                  defaultValue={speaker}
                                  style={{width: 240}}
                                  onChange={handleSpeakerSelect}
                                  options={[
                                      {value: 'aidar', label: "Айдар"},
                                      {value: 'baya', label: "Байя"},
                                      {value: 'kseniya', label: "Ксения"},
                                      {value: 'xenia', label: "Наталья"},
                                      {value: 'eugene', label: "Евгений"},
                                      {value: 'random', label: "случайный"},
                                  ]}
                              />
                          </Space>
                      </Row>
                      <Row className="my-2">
                          <Space>
                              <div className="text-blue-950 text-lg">Качество речи:</div>
                              <Select
                                  defaultValue={sampleRate}
                                  style={{width: 240}}
                                  onChange={handleSampleRateSelect}
                                  options={[
                                      {value: 8000, label: "Низкое (выше скорость генерации)"},
                                      {value: 24000, label: "Среднее"},
                                      {value: 48000, label: "Высокое"},
                                  ]}
                              />
                          </Space>
                      </Row>
                      <div className="flex flex-col items-center my-4">
                          <Button
                              type="primary"
                              onClick={handleGenerateVideo}
                              disabled={(selectedImage === null)}
                          >
                              Сгенерировать видео
                          </Button>
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
