import React, { useEffect, useState } from "react";
import { useParams, useNavigate } from "react-router-dom";
import Carousel from "../components/Carousel";
import "../styles/Display.css";
import { pdfAPI } from "../api/pdf";

interface Metadata {
  timestamp: string;
  audio_timestamps: number[];
  status?: string;
  original_filename: string;
}

const Display = () => {
  const { pdfId } = useParams();
  const navigate = useNavigate();
  const [metadata, setMetadata] = useState<Metadata | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchMetadata = async () => {
      try {
        const response = await fetch(`http://localhost:5000/data/${pdfId}/metadata.json`);
        if (!response.ok) {
          throw new Error('Failed to fetch metadata');
        }
        const data = await response.json();
        if (data.status !== 'completed') {
          navigate(`/configure/${pdfId}`);
          return;
        }
        setMetadata({
          ...data,
          original_filename: data.original_filename
        });
      } catch (error) {
        console.error('Error fetching metadata:', error);
        navigate('/');
      } finally {
        setLoading(false);
      }
    };

    if (pdfId) {
      fetchMetadata();
    }
  }, [pdfId, navigate]);

  if (loading) {
    return <div>Loading...</div>;
  }

  if (!metadata) {
    return <div>Error: Could not load lecture data</div>;
  }

  return (
    <div className="display-container">
      {metadata && (
        <Carousel
          pdfId={pdfId!}
          audioTimestamps={metadata.audio_timestamps}
          timestamp={metadata.original_filename}
        />
      )}
    </div>
  );
};

export default Display;
