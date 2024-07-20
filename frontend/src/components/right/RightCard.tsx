import {Card, Spin} from "antd";
import {LoadingOutlined, YoutubeOutlined} from "@ant-design/icons";

function RightCard(props: {loading: boolean, videoUrl: string | null}) {
    return (
              <Card style={{width: '40%'}}>
                  <h3 className="text-blue-950 text-xl">...</h3>
                  <div className="flex flex-col items-center justify-center max-h-1/5" style={{height: '41vh'}}>
                      {props.loading ? (
                          <Spin indicator={<LoadingOutlined style={{fontSize: 52}} spin/>}/>
                      ) : (
                          <>
                              {props.videoUrl ? (
                                  <div className="h-96">
                                      <video controls className="h-full rounded-lg">
                                          <source src={props.videoUrl} type="video/mp4"/>
                                      </video>
                                  </div>
                              ) : (
                                  <YoutubeOutlined style={{fontSize: '4rem', color: '#ccc', height: '41vh'}}/>
                              )}
                          </>
                      )}
                  </div>
              </Card>
    )
}

export default RightCard