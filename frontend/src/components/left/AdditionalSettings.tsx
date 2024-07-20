import {Collapse} from "antd";
import SelectAttr from "./SelectAttr.tsx";

function AdditionalSettings(props: {
    sampleRate: number,
    handleSetSampleRate: (sampleRate: number) => void,
    enhancer: string,
    handleSetEnhancer: (enhancer: string) => void,
    still: boolean,
    handleSetStill: (still: boolean) => void,
    preprocess: string,
    handleSetPreprocess: (preprocess: string) => void,
}) {

    return (
                              <Collapse
                          size="small"
                          items={[{
                              key: '1', label: 'Дополнительные настройки',
                              children:
                              <div className="flex flex-col">
                                  <SelectAttr
                                      label={"Качество речи"}
                                      defaultValue={props.sampleRate}
                                      onChange={props.handleSetSampleRate}
                                      options={[
                                          {value: 8000, label: "Низкое (выше скорость генерации)"},
                                          {value: 24000, label: "Среднее"},
                                          {value: 48000, label: "Высокое"},
                                      ]}
                                  />
                                  <SelectAttr
                                      label={"Улучшение лица"}
                                      defaultValue={props.enhancer}
                                      onChange={props.handleSetEnhancer}
                                      options={[
                                          {value: "none", label: "Без (выше скорость генерации)"},
                                          {value: "gfpgan", label: "gfpgan"},
                                          {value: "RestoreFormer", label: "RestoreFormer"},
                                      ]}
                                  />
                                  <SelectAttr
                                      label={"Лицо или в полный рост"}
                                      defaultValue={props.still}
                                      onChange={props.handleSetStill}
                                      options={[
                                          {value: false, label: "В полный рост"},
                                          {value: true, label: "Только лицо"},
                                      ]}
                                  />
                                  <SelectAttr
                                      label={"Предварительная обработка"}
                                      defaultValue={props.preprocess}
                                      onChange={props.handleSetPreprocess}
                                      options={[
                                          {value: "crop", label: "crop"},
                                          {value: "extcrop", label: "extcrop"},
                                          {value: "resize", label: "resize"},
                                          {value: "full", label: "full"},
                                          {value: "extfull", label: "extfull"},
                                      ]}
                                  />
                              </div>
                          }]} />
    )
}

export default AdditionalSettings