import React from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { Box, Typography, Button } from '@mui/material';

const ErrorPage = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const { reasoning } = location.state || { reasoning: 'Unknown error occurred' };

  return (
    <Box sx={{ 
      padding: 4,
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      gap: 3
    }}>
      <Typography variant="h4" color="error">
        Non-Lecture PDF Detected
      </Typography>
      
      <Typography variant="body1" sx={{ maxWidth: 600, textAlign: 'center' }}>
        {reasoning}
      </Typography>

      <Button 
        variant="contained" 
        onClick={() => navigate('/')}
        sx={{ mt: 2 }}
      >
        Return to Home
      </Button>
    </Box>
  );
};

export default ErrorPage;