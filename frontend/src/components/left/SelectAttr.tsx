import {Row, Select, Space} from "antd";

function SelectAttr(props: {
    label: string,
    options: {value: string | number | boolean, label: string}[]
    defaultValue: string | number | boolean,
    onChange: (value: any) => void
}) {

    return (
        <Row className="my-2">
          <Space>
              <div className="text-blue-950 text-lg">{props.label}:</div>
              <Select
                  defaultValue={props.defaultValue}
                  style={{width: 300}}
                  onChange={props.onChange}
                  options={props.options}
              />
          </Space>
        </Row>
    )
}

export default SelectAttr