import React from "react";
import { Worker, Viewer } from "@react-pdf-viewer/core";
import { defaultLayoutPlugin } from "@react-pdf-viewer/default-layout";
import "@react-pdf-viewer/core/lib/styles/index.css";
import "@react-pdf-viewer/default-layout/lib/styles/index.css";

interface PDFViewerProps {
  pdfUrl: string | null; // URL of the PDF to display
}

// https://www.youtube.com/watch?v=0FRyKY_PMLE&ab_channel=TheDebugArena

const PDFViewer: React.FC<PDFViewerProps> = ({ pdfUrl }) => {
  const defaultLayoutPluginInstance = defaultLayoutPlugin();

  return (
    <div style={{ paddingTop: "20px", width: "100%", height: "100%" }}>
      {pdfUrl ? (
        <Worker workerUrl={`https://unpkg.com/pdfjs-dist@3.11.174/build/pdf.worker.min.js`}>
          <Viewer
            fileUrl={pdfUrl}
            plugins={[defaultLayoutPluginInstance]}
          />
        </Worker>
      ) : (
        <p>No PDF available</p>
      )}
    </div>
  );
};

export default PDFViewer;
