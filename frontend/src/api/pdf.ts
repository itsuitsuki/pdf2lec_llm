import axios from 'axios';
import { PDFFile, GenerateOptions } from './types';

const API_BASE_URL = 'http://localhost:5000/api/v1';

// 创建一个 axios 实例，设置通用配置
const axiosInstance = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  withCredentials: true, // 允许跨域请求携带凭证
  headers: {
    'Content-Type': 'application/json',
  },
});

// 添加请求拦截器
axiosInstance.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

export const pdfAPI = {
  uploadSlide: async (formData: FormData) => {
    console.log('=== API: uploadSlide called ===');
    try {
      const response = await axiosInstance.post('/upload-slide', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      console.log('🟢 API Response:', response);
      return response;
    } catch (error: any) {
      console.error('🔴 API Error:', error);
      throw error;
    }
  },

  getPDFs: async (type: 'slide' | 'textbook'): Promise<{ data: PDFFile[] }> => {
    console.log(`=== API: getPDFs(${type}) called ===`);
    try {
      const response = await axiosInstance.get<PDFFile[]>(`/pdfs/${type}`);
      // 处理每个文件的metadata以获取原始文件名
      const processedFiles = response.data.map(file => ({
        ...file,
        displayName: file.metadata?.original_filename || file.filename
      }));
      return { data: processedFiles };
    } catch (error: any) {
      console.error('🔴 Failed to get PDFs:', error);
      throw error;
    }
  },

  deletePDF: async (type: 'slide' | 'textbook', id: string) => {
    try {
      const response = await axiosInstance.delete(`/pdfs/${type}/${id}`);
      return response;
    } catch (error: any) {
      if (error.response) {
        throw error.response;
      }
      throw error;
    }
  },

  uploadTextbook: async (formData: FormData, slideId: string) => {
    try {
      const response = await axiosInstance.post(`/upload-textbook/${slideId}`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      return response;
    } catch (error) {
      console.error('Failed to upload textbook:', error);
      throw error;
    }
  },

  generateLecture: async (options: GenerateOptions) => {
    try {
      const response = await axiosInstance.post('/lec_generate', options);
      return response;
    } catch (error) {
      console.error('Failed to generate lecture:', error);
      throw error;
    }
  },

  getTaskStatus: async (taskId: string) => {
    try {
      const response = await axiosInstance.get(`/task_status/${taskId}`);
      return response;
    } catch (error) {
      console.error('Failed to get task status:', error);
      throw error;
    }
  },
};