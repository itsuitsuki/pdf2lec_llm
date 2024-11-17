import axios from 'axios';
import { PDFFile } from './types';

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
      console.log('🟢 PDF list received:', response.data);
      return response;
    } catch (error: any) {
      console.error('🔴 Failed to get PDFs:', error);
      throw error;
    }
  },

  deletePDF: async (type: 'slide' | 'textbook', id: string) => {
    try {
      const response = await axios.delete(`${API_BASE_URL}/pdfs/${type}/${id}`);
      return response;
    } catch (error: any) {
      if (error.response) {
        throw error.response;
      }
      throw error;
    }
  },
};