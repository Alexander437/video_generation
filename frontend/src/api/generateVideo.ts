

// const enum availableSampleRate {
//     Low = 8000,
//     Middle = 24000,
//     High = 48000,
// }

import axiosInstance from "./axiosConfig.ts";

type GenerateVideoParams = {
    imgFile: File;
    text: string;
    speaker: string;
    sampleRate: number;
    enhancer: string;
    still: boolean;
    preprocess: string;
}

export const generateVideo = async (params: GenerateVideoParams): Promise<Blob> => {
    const { imgFile, text, speaker, sampleRate ,
    enhancer, still, preprocess} = params;

    const formData = new FormData();
    formData.append('img_file', imgFile);

    const queryParams = new URLSearchParams({
        text,
        speaker,
        enhancer,
        preprocess,
        still: still.toString(),
        sample_rate: sampleRate.toString(),
    });

    const url = `/video/?${queryParams.toString()}`;

    const response = await axiosInstance.post(url, formData, {
        responseType: "blob",
        headers: {
            "Content-Type": "multipart/form-data",
        }
    });

    return response.data;
};