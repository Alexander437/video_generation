import {Button, Card, Input, type UploadFile, type UploadProps} from "antd";
import UploadImage from "./UploadImage.tsx";
import React, {useState} from "react";
import {generateVideo} from "../../api/generateVideo.ts";
import SelectAttr from "./SelectAttr.tsx";
import AdditionalSettings from "./AdditionalSettings.tsx";

const { TextArea } = Input;


function LeftCard(props: {
    setVideoUrl: (videoUrl: string) => void,
    setLoading: (loading: boolean) => void

}) {
    const [fileList, setFileList] = useState<UploadFile[]>([]);
    const [selectedImage, setSelectedImage] = useState<File | null>(null);
    const [inputText, setInputText] = useState<string>("Привет");
    const [speaker, setSpeaker] = useState<string>("aidar");

    const [sampleRate, setSampleRate] = useState<number>(24000);
    const [enhancer, setEnhancer] = useState<string>("gfpgan");
    const [still, setStill] = useState<boolean>(false);
    const [preprocess, setPreprocess] = useState<string>("crop");


    const handleSetSpeaker = (value: string) => setSpeaker(value)
    const handleInputChange = (text: any) => setInputText(text.target.value)

    const handleSetSampleRate = (value: number) => setSampleRate(value)
    const handleSetEnhancer = (value: string) => setEnhancer(value)
    const handleSetStill = (value: boolean) => setStill(value);
    const handleSetPreprocess = (value: string) => setPreprocess(value)

    const handlePreview = async (file: UploadFile) => {
        const fileObj = file.originFileObj || (await fetch(file.url).then(r => r.blob()));
        setSelectedImage(fileObj as File);
        console.log(selectedImage)
    };

    const generateImage = async (event: React.FormEvent) => {
        event.preventDefault();
        props.setLoading(true);

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
            props.setLoading(false);
        }
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

    const handleGenerateVideo = async () => {
        if (selectedImage) {
            try {
                props.setLoading(true);
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
                props.setVideoUrl(videoUrl);
                props.setLoading(false);
            }
            catch (error) {
                console.error(error);
            }
        }
    }

    return (
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
                      <SelectAttr
                          label={"Голос"}
                          defaultValue={speaker}
                          onChange={handleSetSpeaker}
                          options={[
                              {value: 'aidar', label: "Айдар"},
                              {value: 'baya', label: "Байя"},
                              {value: 'kseniya', label: "Ксения"},
                              {value: 'xenia', label: "Наталья"},
                              {value: 'eugene', label: "Евгений"},
                              {value: 'random', label: "случайный"},
                          ]}
                      />
                      <AdditionalSettings
                          sampleRate={sampleRate}
                          handleSetSampleRate={handleSetSampleRate}
                          enhancer={enhancer}
                          handleSetEnhancer={handleSetEnhancer}
                          still={still}
                          handleSetStill={handleSetStill}
                          preprocess={preprocess}
                          handleSetPreprocess={handleSetPreprocess}
                      />
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
    )
}

export default LeftCard