import React from "react";
import Box from "@mui/material/Box";
import Sidebar from "../components/Sidebar";

const Starred = () => {
  return (
    <div>
      <Box sx={{ display: "flex", paddingTop: "20px" }}>
        <Sidebar />
        <h1>Starred</h1>
      </Box>
    </div>
  );
};

export default Starred;
