import React from "react";
import Box from "@mui/material/Box";
import Sidebar from "../components/Sidebar";

const Records = () => {
  return (
    <div>
      <Box sx={{ display: "flex", paddingTop: "20px" }}>
        <Sidebar />
        <h1>My Records</h1>
      </Box>
    </div>
  );
};

export default Records;
