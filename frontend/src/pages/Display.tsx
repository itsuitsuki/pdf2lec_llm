import React from "react";
import { useParams } from "react-router-dom";
import Carousel from "../components/Carousel";
import "../styles/Display.css";

const Display = () => {
  const { pdfId } = useParams();
  const pdfUrl = `http://localhost:8080/files/${pdfId}`;
  const gridContent = Array.from({ length: 9 }, (_, index) => index + 1); // Create an array with values 1 to 9

  return (
    <div className="display-container">
      <Carousel
        pdfUrl={pdfUrl}
        audioUrl="../../public/L6_Classification_slide1.mp3"
      />
      {/* <div className="grid-container">
        {gridContent.map((item) => (
          <div className="grid-item" key={item}>
            {item}
          </div>
        ))}
      </div> */}
    </div>
  );
};

export default Display;
