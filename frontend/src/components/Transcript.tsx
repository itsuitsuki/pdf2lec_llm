import React, { useEffect, useState } from "react";
import { useParams } from "react-router-dom";
import { useAuth } from "../contexts/AuthContext";
import { getAuthHeaders } from "../utils/auth";

interface TranscriptProps {
  currentSlide: number;
  pdfId: string;
}

const Transcript: React.FC<TranscriptProps> = ({ currentSlide, pdfId }) => {
  const [files, setFiles] = useState<string[]>([]);
  const [transcript, setTranscript] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    const loadTranscript = async () => {
      // Check if files are available and currentSlide is within range
      if (
        files.length === 0 ||
        currentSlide < 0 ||
        currentSlide >= files.length
      ) {
        setTranscript("No valid file selected.");
        return;
      }

      const slideURL = files[currentSlide];
      const transcriptURL = `http://localhost:8000/data/${pdfId}/generated_texts/lecture/${slideURL}`;

      // Get the token from localStorage or context
      const token = localStorage.getItem("token");

      try {
        // Fetch the text file with the Authorization header
        const response = await fetch(transcriptURL, {
          headers: {
            Authorization: `Bearer ${token}`,
          },
        });

        if (!response.ok) {
          throw new Error("Failed to load transcript.");
        }

        // Convert the response to text
        const transcriptText = await response.text();

        // Set the transcript state with the fetched text
        setTranscript(transcriptText);
      } catch (error) {
        console.error("Error loading transcript:", error);
        setTranscript("Error loading transcript.");
      }
    };

    loadTranscript();
  }, [files, currentSlide, pdfId]); // Re-run the effect when files, currentSlide, or pdfId change

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
      {files ? <div>{transcript}</div> : <p>Loading transcript...</p>}
    </div>
  );
};

export default Transcript;
