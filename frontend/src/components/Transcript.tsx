import React, { useEffect, useState } from "react";
import { useParams } from "react-router-dom";
import { useAuth } from "../contexts/AuthContext";
import { getAuthHeaders } from "../utils/auth";

const Transcript = ({ currentTime }: { currentTime: number }) => {
  const { pdfId } = useParams<{ pdfId: string }>();
  const [files, setFiles] = useState<string[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchTranscriptFiles = async () => {
      try {
        const response = await fetch(
          "http://localhost:8000/api/v1/retrieve_filenames",
          {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
              Authorization: `Bearer ${localStorage.getItem("token")}`,
            },
            body: JSON.stringify({ pdfId }), // Send pdfId in the request body
          }
        );

        if (!response.ok) {
          throw new Error("Failed to fetch transcript files");
        }

        const data = await response.json();
        console.log("FILENAMES", data);
        setFiles(data.files); // Assuming the response contains a 'files' array
      } catch (error) {
        console.error("Error fetching transcript files:", error);
        setError("Error fetching transcript files");
      } finally {
        setLoading(false);
      }
    };
    if (pdfId) {
      fetchTranscriptFiles();
    }
  }, [pdfId]);

  return (
    <div className="transcript-container">
      {files ? <div>{files}</div> : <p>Loading transcript...</p>}
    </div>
  );
};

export default Transcript;
