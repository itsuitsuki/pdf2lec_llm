import React, { useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { Box, Button, TextField, FormControlLabel, Switch, Typography } from '@mui/material';
import { pdfAPI } from '../api/pdf';

const ConfigurePage = () => {
  const { pdfId } = useParams();
  const navigate = useNavigate();
  const [textbookFile, setTextbookFile] = useState<File | null>(null);
  const [complexity, setComplexity] = useState<number>(2);
  const [useRAG, setUseRAG] = useState<boolean>(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    try {
      // First upload textbook if provided
      let textbookName = null;
      if (textbookFile && useRAG) {
        const formData = new FormData();
        formData.append("file", textbookFile);
        const response = await pdfAPI.uploadTextbook(formData);
        textbookName = response.data.filename;
      }

      // Then start generation
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

      // Navigate back to home page after starting generation
      navigate('/');
    } catch (error) {
      console.error('Generation failed:', error);
      alert('Failed to start generation');
    }
  };

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
          />

          <Button 
            variant="contained" 
            type="submit"
            sx={{ mt: 2 }}
          >
            Start Generation
          </Button>
        </Box>
      </form>
    </Box>
  );
};

export default ConfigurePage;