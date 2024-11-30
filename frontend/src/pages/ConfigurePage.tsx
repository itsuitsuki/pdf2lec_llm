import React, { useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { Box, Button, TextField, FormControlLabel, Switch, Typography } from '@mui/material';
import { pdfAPI } from '../api/pdf';

const ConfigurePage = () => {
  const { pdfId } = useParams<{ pdfId: string }>();
  const navigate = useNavigate();
  const [textbookFile, setTextbookFile] = useState<File | null>(null);
  const [complexity, setComplexity] = useState<number>(2);
  const [useRAG, setUseRAG] = useState<boolean>(false);
  const [isUploading, setIsUploading] = useState<boolean>(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!pdfId) {
      alert('Invalid PDF ID');
      return;
    }

    setIsUploading(true);
    try {
      // 上传教科书（如果有）
      let textbookName = null;
      if (textbookFile && useRAG) {
        const formData = new FormData();
        formData.append("file", textbookFile);
        const response = await pdfAPI.uploadTextbook(formData, pdfId);
        textbookName = response.data.metadata.textbook_filename;
      }

      // 开始生成
      const response = await pdfAPI.generateLecture({
        similarity_threshold: 0.4,
        text_generating_context_size: 2,
        max_tokens: 1000,
        pdf_name: pdfId,
        page_model: "gpt-4o",
        digest_model: "gpt-4o-mini",
        tts_model: "tts-1",
        tts_voice: "alloy",
        complexity: complexity,
        debug_mode: false,
        use_rag: useRAG,
        textbook_name: textbookName,
      });

      // 存储任务ID
      localStorage.setItem('lastGenerationTask', response.data.task_id);
      
      // 导航到主页并触发立即轮询
      navigate('/', { state: { shouldPoll: true } });
    } catch (error) {
      console.error('Generation failed:', error);
      alert('Failed to start generation. Please try again.');
    } finally {
      setIsUploading(false);
    }
  };

  if (!pdfId) {
    return (
      <Box sx={{ padding: 4, textAlign: 'center' }}>
        <Typography variant="h5" color="error">
          Invalid PDF ID. Please select a PDF from the home page.
        </Typography>
      </Box>
    );
  }

  return (
    <Box sx={{ padding: 4, maxWidth: 600, margin: '0 auto' }}>
      <Typography variant="h4" gutterBottom>
        Configure Generation
      </Typography>
      
      <form onSubmit={handleSubmit}>
        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
          <FormControlLabel
            control={
              <Switch
                checked={useRAG}
                onChange={(e) => setUseRAG(e.target.checked)}
              />
            }
            label="Use Textbook Reference"
          />

          {useRAG && (
            <Box>
              <Typography variant="subtitle1" gutterBottom>
                Upload Textbook (PDF)
              </Typography>
              <input
                type="file"
                accept=".pdf"
                onChange={(e) => setTextbookFile(e.target.files?.[0] || null)}
                disabled={isUploading}
              />
            </Box>
          )}

          <TextField
            label="Complexity Level"
            type="number"
            value={complexity}
            onChange={(e) => setComplexity(Number(e.target.value))}
            inputProps={{ min: 1, max: 3 }}
            helperText="1: Brief, 2: Default, 3: Detailed"
            disabled={isUploading}
          />

          <Button 
            variant="contained" 
            type="submit"
            disabled={isUploading}
            sx={{ mt: 2 }}
          >
            {isUploading ? 'Processing...' : 'Start Generation'}
          </Button>
        </Box>
      </form>
    </Box>
  );
};

export default ConfigurePage;