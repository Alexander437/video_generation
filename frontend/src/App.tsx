import React, {useState} from "react";
import {generateVideo} from "./api/generateVideo.ts";
import {Button, Card, Collapse, Input, Row, Select, Space, Spin, type UploadFile, type UploadProps} from "antd";
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
    const [enhancer, setEnhancer] = useState<string | null>(null);
    const [still, setStill] = useState<boolean | null>(null);
    const [preprocess, setPreprocess] = useState<string | null>(null);

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

    const generateImage = async (event: React.FormEvent) => {
        event.preventDefault();
        setLoading(true);

        try {
            const response = await fetch('http://127.0.0.1:8000/avatar/', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
            });

            if (!response.ok) {
                throw new Error('Network response was not ok');
            }

            const blob = await response.blob();
            const name = response.headers.get('Content-Disposition')?.split('filename=')[1] || 'avatar.jpg';
            const file = new File([blob], name, { type: blob.type });
            setSelectedImage(file);
            console.log(file);

        } catch (error) {
            console.error('Error generating avatar:', error);
        } finally {
            setLoading(false);
        }
    };

  const handlePreview = async (file: UploadFile) => {
      const fileObj = file.originFileObj || (await fetch(file.url).then(r => r.blob()));
      setSelectedImage(fileObj as File);
      console.log(selectedImage)
  };

    const handleSpeakerSelect = (value: string) => setSpeaker(value)
    const handleInputChange = (text: any) => setInputText(text.target.value)

    const handleSampleRateSelect = (value: number) => setSampleRate(value)
    const handleEnhancer = (value: string) => setEnhancer(value)
    const handleStill = (value: boolean) => setStill(value);
    const handlePreprocess = (value: string) => setPreprocess(value)



    const handleGenerateVideo = async () => {
        if (selectedImage) {
            try {
                setLoading(true);
                const videoBlob = await generateVideo({
                    imgFile: selectedImage,
                    text: inputText,
                    speaker: speaker,
                    sampleRate: sampleRate,
                    enhancer: enhancer,
                    still: still,
                    preprocess: preprocess,
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
                      generateImage={generateImage}
                      selectedImage={selectedImage}
                  />
                      <div className="flex flex-col items-center text-blue-950 text-lg">Текст:</div>
                      <TextArea defaultValue={inputText} onChange={handleInputChange} style={{width: '1000'}}/>
                      <Row className="my-2">
                          <Space>
                              <div className="text-blue-950 text-lg">Голос:</div>
                              <Select
                                  defaultValue={speaker}
                                  style={{width: 300}}
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
                      <Collapse
                          size="small"
                          items={[{
                              key: '1', label: 'Дополнительные настройки',
                              children:
                              <div className="flex flex-col">
                                  <Row className="my-2">
                                      <Space>
                                          <div className="text-blue-950 text-base">Качество речи:</div>
                                          <Select
                                              defaultValue={sampleRate}
                                              style={{width: 306}}
                                              onChange={handleSampleRateSelect}
                                              options={[
                                                  {value: 8000, label: "Низкое (выше скорость генерации)"},
                                                  {value: 24000, label: "Среднее"},
                                                  {value: 48000, label: "Высокое"},
                                              ]}
                                          />
                                      </Space>
                                  </Row>
                                  <Row className="my-2">
                                      <Space>
                                          <div className="text-blue-950 text-base">Улучшение лица:</div>
                                          <Select
                                              defaultValue={"gfpgan"}
                                              style={{width: 290}}
                                              onChange={handleEnhancer}
                                              options={[
                                                  {value: "none", label: "Без (выше скорость генерации)"},
                                                  {value: "gfpgan", label: "gfpgan"},
                                                  {value: "RestoreFormer", label: "RestoreFormer"},
                                              ]}
                                          />
                                      </Space>
                                  </Row>
                                  <Row className="my-2">
                                      <Space>
                                          <div className="text-blue-950 text-base">Лицо или в полный рост:</div>
                                          <Select
                                              defaultValue={false}
                                              style={{width: 234}}
                                              onChange={handleStill}
                                              options={[
                                                  {value: false, label: "В полный рост"},
                                                  {value: true, label: "Только лицо"},
                                              ]}
                                          />
                                      </Space>
                                  </Row>
                                  <Row className="my-2">
                                      <Space>
                                          <div className="text-blue-950 text-base">Предварительная обработка:</div>
                                          <Select
                                              defaultValue={"crop"}
                                              style={{width: 200}}
                                              onChange={handlePreprocess}
                                              options={[
                                                  {value: "crop", label: "crop"},
                                                  {value: "extcrop", label: "extcrop"},
                                                  {value: "resize", label: "resize"},
                                                  {value: "full", label: "full"},
                                                  {value: "extfull", label: "extfull"},
                                              ]}
                                          />
                                      </Space>
                                  </Row>
                              </div>
                          }]} />
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
                  {/*<h3 className="text-blue-950 text-xl">Видео:</h3>*/}
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
