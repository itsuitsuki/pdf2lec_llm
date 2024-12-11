import React, { useEffect, useState } from "react";
import Sidebar from "../components/Sidebar";
import Box from "@mui/material/Box";
import Dialog from "@mui/material/Dialog";
import DialogTitle from "@mui/material/DialogTitle";
import DialogContent from "@mui/material/DialogContent";
import DialogActions from "@mui/material/DialogActions";
import Button from "@mui/material/Button";
import UploadIcon from "@mui/icons-material/CloudUpload"; // MUI upload icon

import List from "@mui/material/List";
import ListItem from "@mui/material/ListItem";
import ListItemAvatar from "@mui/material/ListItemAvatar";
import ListItemText from "@mui/material/ListItemText";
import Avatar from "@mui/material/Avatar";
import IconButton from "@mui/material/IconButton";
import FolderIcon from "@mui/icons-material/Folder";
import DeleteIcon from "@mui/icons-material/Delete";

import "../styles/Home.css";
import { pdfAPI } from '../api/pdf';
import { useNavigate } from "react-router-dom";
import { PDFFile } from '../api/types';

const StatusIndicator: React.FC<{ status: string }> = ({ status }) => {
  const getStatusColor = () => {
    switch (status) {
      case 'completed':
        return '#4CAF50';
      case 'generating':
        return '#FFC107';
      case 'pending':
        return '#9E9E9E';
      case 'failed':
        return '#F44336';
      default:
        return '#9E9E9E';
    }
  };

  const getStatusText = () => {
    switch (status) {
      case 'completed':
        return '✅ Ready';
      case 'generating':
        return '🔄 Generating...';
      case 'pending':
        return '⚪ Not Started';
      case 'failed':
        return '❌ Failed';
      case 'non_lecture_pdf_error':
        return '❌ Non-Lecture PDF';
      default:
        return 'Unknown Error';
    }
  };

  return (
    <div style={{
      display: 'flex',
      alignItems: 'center',
      gap: '8px',
      color: getStatusColor(),
      fontSize: '0.875rem'
    }}>
      <span style={{
        width: '8px',
        height: '8px',
        borderRadius: '50%',
        backgroundColor: getStatusColor()
      }} />
      {getStatusText()}
    </div>
  );
};

const Home = () => {
  const [open, setOpen] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [slides, setSlides] = useState<PDFFile[]>([]);
  const [selectedPDF, setSelectedPDF] = useState<string | null>(null);
  const navigate = useNavigate();
  const [pollingErrors, setPollingErrors] = useState(0);
  const MAX_POLLING_ERRORS = 3;

  useEffect(() => {
    console.log('Home component mounted');
    loadSlides();
  }, []);

  useEffect(() => {
    const pollStatus = async () => {
      try {
        const pendingOrGeneratingSlides = slides.filter(slide => 
          ['pending', 'generating'].includes(slide.metadata?.status || '')
        );
        
        if (pendingOrGeneratingSlides.length > 0) {
          await loadSlides();
          setPollingErrors(0); // Reset error count on successful poll
        }
      } catch (error) {
        console.error('Polling error:', error);
        setPollingErrors(prev => prev + 1);
        
        if (pollingErrors >= MAX_POLLING_ERRORS) {
          console.error('Max polling errors reached, stopping polling');
          return;
        }
      }
    };

    const intervalId = setInterval(pollStatus, 2000); // Poll every 2 seconds

    // Cleanup function
    return () => {
      clearInterval(intervalId);
    };
  }, [slides, pollingErrors]);

  const loadSlides = async () => {
    console.log('=== Starting to load slides ===');
    try {
      const response = await pdfAPI.getPDFs('slide');
      console.log('🟢 Slides loaded successfully:', response.data);
      if (response.data) {
        // 按上传时间排序，最新的在前
        const sortedSlides = response.data.sort((a, b) => {
          const timeA = new Date(a.metadata?.upload_time || 0).getTime();
          const timeB = new Date(b.metadata?.upload_time || 0).getTime();
          return timeB - timeA;  // 降序排列
        });
        setSlides(sortedSlides);
        console.log('🟢 Slides state updated:', sortedSlides);
      }
    } catch (error) {
      console.error('🔴 Failed to fetch slides:', error);
    }
  };

  const handleClickOpen = () => {
    setOpen(true);
  };

  const handleClose = () => {
    setOpen(false);
    setFile(null);
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0];
    if (selectedFile) {
      setFile(selectedFile);
    }
  };

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    console.log('=== Starting file upload ===');
    
    if (!file) {
      console.warn('⚠️ No file selected');
      alert("Please select a PDF file.");
      return;
    }

    try {
      console.log('📁 Selected file:', file.name);
      const formData = new FormData();
      formData.append("file", file);

      const response = await pdfAPI.uploadSlide(formData);
      console.log('🟢 Upload successful:', response);
      
      await loadSlides();
      handleClose();
      alert('Slide uploaded successfully!');
    } catch (error: any) {
      console.error('🔴 Upload failed:', error);
      console.error('Error details:', {
        message: error?.message,
        response: error?.response,
        status: error?.response?.status,
      });
      alert(`Upload failed: ${error?.response?.data?.detail || error.message || 'Unknown error'}`);
    }
  };

  const handleDelete = async (pdfId: string) => {
    try {
      await pdfAPI.deletePDF('slide', pdfId);
      loadSlides(); // Refresh the list after deletion
    } catch (error) {
      console.error('Delete failed:', error);
      alert('Failed to delete the slide');
    }
  };

  const showPDF = (pdf: PDFFile) => {
    switch (pdf.metadata?.status) {
      case 'completed':
        navigate(`/display/${pdf.id}`);
        break;
      case 'generating':
        alert('This lecture is currently being generated. Please wait for it to complete.');
        break;
      case 'non_lecture_pdf_error':
        navigate('/error', { 
          state: { 
            reasoning: pdf.metadata?.validation?.reasoning || 'Unknown error occurred'
          }
        });
        break;
      case 'failed':
        alert('Generation failed. Please try upload another PDF');
        navigate(`/configure/${pdf.id}`);
        break;
      case 'pending':
      default:
        navigate(`/configure/${pdf.id}`);
        break;
    }
  };

  return (
    <div>
      <Box sx={{ display: "flex", paddingTop: "20px" }}>
        <Sidebar />
        <Box
          sx={{
            padding: "20px",
            display: "flex",
            flexGrow: 1,
            flexDirection: "column",
          }}
        >
          <h1>Home</h1>
          <Box
            onClick={handleClickOpen}
            sx={{
              marginTop: "20px",
              width: 200,
              height: 200,
              border: "2px dashed #aaa",
              borderRadius: "10px",
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              justifyContent: "center",
              cursor: "pointer",
              backgroundColor: "#f9f9f9",
              transition: "background-color 0.2s, border-color 0.2s",
              "&:hover": {
                backgroundColor: "#8FD3F8",
                borderColor: "#83D8FF",
                "& .upload-icon": {
                  color: "#fff", // Change icon color to white on hover
                },
                "& .upload-purpose": {
                  color: "#fff", // Change icon color to white on hover
                },
              },
            }}
          >
            <UploadIcon
              className="upload-icon"
              sx={{ fontSize: 50, color: "#555", transition: "color 0.3s" }}
            />
            <div className="upload-purpose">Upload PDF</div>
          </Box>

          {/* Dialog Modal with Form */}
          <Dialog open={open} onClose={handleClose}>
            <DialogTitle>Upload Slide PDF</DialogTitle>
            <DialogContent>
              <form className="form-container" onSubmit={handleSubmit}>
                <input
                  type="file"
                  className="form-control"
                  accept="application/pdf"
                  onChange={handleFileChange}
                  required
                />
                <DialogActions>
                  <Button onClick={handleClose}>Cancel</Button>
                  <Button type="submit">Upload</Button>
                </DialogActions>
              </form>
            </DialogContent>
          </Dialog>

          <Box
            sx={{
              paddingTop: "60px",
              display: "flex",
              flexDirection: "column",
            }}
          >
            <h1>Uploaded PDFs</h1>
            <div className="uploaded-container">
              <List sx={{ width: "45%" }}>
                {slides && slides.length > 0 ? (
                  slides.map((pdf) => (
                    <div className="pdf-container" key={pdf.id}>
                      <ListItem sx={{ 
                        display: "flex", 
                        alignItems: "center",
                        justifyContent: "space-between",
                        padding: "8px 16px"
                      }}>
                        <div style={{ display: "flex", alignItems: "center", flex: 1}}>
                          <ListItemAvatar>
                            <Avatar sx={{ backgroundColor: "rgb(95, 95, 226)", color: "white" }}>
                              <FolderIcon />
                            </Avatar>
                          </ListItemAvatar>
                          <ListItemText
                            sx={{ cursor: "pointer" }}
                            onClick={() => showPDF(pdf)}
                            primary={<h3>{pdf.displayName}</h3>}
                          />
                        </div>
                        <div style={{ display: "flex", alignItems: "center", gap: "16px" }}>
                          <StatusIndicator status={pdf.metadata?.status || 'pending'} />
                          <IconButton 
                            edge="end" 
                            aria-label="delete"
                            onClick={(e) => {
                              e.stopPropagation();
                              handleDelete(pdf.id);
                            }}
                          >
                            <DeleteIcon />
                          </IconButton>
                        </div>
                      </ListItem>
                    </div>
                  ))
                ) : (
                  <ListItem>
                    <ListItemText primary="No PDFs uploaded yet" />
                  </ListItem>
                )}
              </List>
            </div>
          </Box>
        </Box>
      </Box>
    </div>
  );
};

export default Home;
