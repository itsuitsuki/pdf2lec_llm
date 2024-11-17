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
import PDFViewer from "../components/pdfViewer";
import { useNavigate } from "react-router-dom";
import { PDFFile } from '../api/types';

const Home = () => {
  const [open, setOpen] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [slides, setSlides] = useState<PDFFile[]>([]);
  const [selectedPDF, setSelectedPDF] = useState<string | null>(null);
  const navigate = useNavigate();

  useEffect(() => {
    console.log('Home component mounted');
    loadSlides();
  }, []);

  const loadSlides = async () => {
    console.log('=== Starting to load slides ===');
    try {
      const response = await pdfAPI.getPDFs('slide');
      console.log('🟢 Slides loaded successfully:', response.data);
      if (response.data) {
        setSlides(response.data);
        console.log('🟢 Slides state updated:', response.data);
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
    setSelectedPDF(pdf.path);
    navigate(`/pdf/${pdf.id}`);
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
              <List sx={{ width: "40%" }}>
                {slides && slides.length > 0 ? (
                  slides.map((pdf) => (
                    <div className="pdf-container" key={pdf.id}>
                      <ListItem sx={{ display: "flex", alignItems: "center" }}>
                        <ListItemAvatar>
                          <Avatar sx={{ backgroundColor: "rgb(95, 95, 226)", color: "white" }}>
                            <FolderIcon />
                          </Avatar>
                        </ListItemAvatar>
                        <ListItemText
                          sx={{ cursor: "pointer" }}
                          onClick={() => showPDF(pdf)}
                          primary={<h3>{pdf.filename.split('_').slice(1).join('_')}</h3>}
                        />
                        <IconButton 
                          edge="end" 
                          aria-label="delete"
                          onClick={() => handleDelete(pdf.id)}
                        >
                          <DeleteIcon />
                        </IconButton>
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

          {/* Render the PDF viewer conditionally */}
          {/* <Box
            sx={{
              display: "flex",
              flexDirection: "row",
              // justifyContent: "center",
            }}
          >
            {selectedPDF && (
              <div className="pdf-viewer">
                <PDFViewer pdfUrl={selectedPDF} />
              </div>
            )}
            {selectedPDF && (
              <div style = {{display:"flex", flexDirection:"column", paddingLeft: "20px", gap: "1rem", paddingTop: "60px"}}>
                <button onClick={playAudio} style ={{backgroundColor: "#8FD3F8"}}>Play</button>
                <button onClick={pause} style ={{backgroundColor: "#8FD3F8"}}>Pause</button>
              </div>
            )}
          </Box> */}
        </Box>
      </Box>
    </div>
  );
};

export default Home;
