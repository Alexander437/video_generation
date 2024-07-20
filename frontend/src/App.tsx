import {useState} from "react";
import { Row } from "antd";
import LeftCard from "./components/left/LeftCard.tsx";
import RightCard from "./components/right/RightCard.tsx";


function App() {
    const [videoUrl, setVideoUrl] = useState<string | null>(null);
    const [loading, setLoading] = useState<boolean>(false);


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
              <LeftCard
                  setVideoUrl={setVideoUrl}
                  setLoading={setLoading}
              />
              {/* Правая часть экрана */}
              <RightCard loading={loading} videoUrl={videoUrl}/>
          </Row>
      </>
)
}

export default App
