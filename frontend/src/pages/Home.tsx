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
import axios from "axios";
// import { PDFViewer } from "@react-pdf/renderer";
import PDFViewer from "../components/pdfViewer";
import { useNavigate } from "react-router-dom";


interface PDF {
  pdf: string;
  title: string;
}

const Home = () => {
  const [open, setOpen] = useState(false);
  const [title, setTitle] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [allPDFS, setAllPDFS] = useState<PDF[]>([]);
  const [selectedPDF, setSelectedPDF] = useState<string | null>(null); // State to manage selected PDF
  const navigate = useNavigate();

  useEffect(() => {
    getPDF();
  }, []);

  // let audio = new Audio("../../public/L6_Classification_slide1.mp3");

  // const playAudio = () => {
  //   audio.play();
  // };

  // const pause = () => {
  //   audio.pause();
  // };

  const getPDF = async () => {
    const result = await axios.get("http://localhost:8080/get-files");
    console.log(result.data.data);
    setAllPDFS(result.data.data);
  };

  const handleClickOpen = () => {
    setOpen(true);
  };

  const handleClose = () => {
    setOpen(false);
    setFile(null);
    setTitle("");
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0]; // Check if files exist and select the first one
    if (selectedFile) {
      setFile(selectedFile);
    }
  };

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault(); // Prevent default form submission
    if (!file) {
      alert("Please upload a PDF file.");
      return;
    }
    const formData = new FormData();
    formData.append("title", title);
    formData.append("file", file);
    console.log("Title:", title);
    console.log("File:", file);
    const result = await axios.post(
      "http://localhost:8080/upload-files",
      formData,
      {
        headers: { "Content-Type": "multipart/form-data" },
      }
    );
    getPDF();
    alert(`Uploaded ${title} successfully!`);
    console.log(result);

    handleClose();
  };

  const showPDF = (pdf: string) => {
    const url = `http://localhost:8080/files/${pdf}`;
    console.log(`http://localhost:8080/files/${pdf}`);
    setSelectedPDF(`http://localhost:8080/files/${pdf}`); // Update state to the selected PDF
    navigate(`/pdf/${pdf}`);
  };

  // Remove the PDF from the state
  const handleDelete = (pdfPath: string) => {
    setAllPDFS((prevPDFS) => prevPDFS.filter((data) => data.pdf !== pdfPath));
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
            <DialogTitle>Upload PDF</DialogTitle>
            <DialogContent>
              <form className="form-container" onSubmit={handleSubmit}>
                <input
                  type="text"
                  className="form-control"
                  placeholder="Title"
                  onChange={(e) => setTitle(e.target.value)}
                  required
                />
                <input
                  type="file"
                  className="form-control"
                  accept="application/pdf"
                  onChange={handleFileChange}
                  required
                />
                <DialogActions>
                  <Button onClick={handleClose}>Cancel</Button>
                  <Button type="submit">Submit</Button>
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
                {allPDFS &&
                  allPDFS.map((data) => (
                    <div className="pdf-container">
                      <ListItem
                        key={data.pdf}
                        sx={{ display: "flex", alignItems: "center" }}
                      >
                        <ListItemAvatar>
                          <Avatar
                            sx={{
                              backgroundColor: "rgb(95, 95, 226)",
                              color: "white",
                            }}
                          >
                            <FolderIcon />
                          </Avatar>
                        </ListItemAvatar>
                        <ListItemText
                          sx={{ cursor: "pointer" }}
                          onClick={() => showPDF(data.pdf)}
                          primary={
                            <div>
                              <h3>{data.title}</h3>
                            </div>
                          }
                        />
                        <IconButton edge="end" aria-label="delete">
                          <DeleteIcon onClick={() => handleDelete(data.pdf)} />
                        </IconButton>
                      </ListItem>
                    </div>
                  ))}
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
