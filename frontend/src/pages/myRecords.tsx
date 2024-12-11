import React from "react";
import Box from "@mui/material/Box";
import Sidebar from "../components/Sidebar";
import "../styles/MyRecords.css";

interface TagProp {
  title: string;
  tags: string[];
  description: string;
  total_time: number;
  time_uploaded: number;
}

const Record: React.FC<TagProp> = ({
  title,
  tags,
  description,
  total_time,
  time_uploaded,
}) => {
  return (
    <div className="record-container">
      <div className="tag-title">{title}</div>
      <div className="tag-container">
        {tags.map((tag, index) => (
          <div className="tags" key={index}>
            {tag}
          </div>
        ))}
      </div>
      <div className="record-description">{description}</div>
      <div className="record-footer">
        <span>Total Time: {total_time} mins</span>
        <span>Uploaded: {new Date(time_uploaded).toLocaleString()}</span>
      </div>
    </div>
  );
};
const MyRecords = () => {
  // Example data for demonstration
  const all_records = [
    {
      title: "PDF Lecture Notes",
      tags: ["GPT", "RAG-Compressed", "AI"],
      description: "A detailed lecture on AI and its advancements.",
      total_time: 90,
      time_uploaded: Date.now(),
    },
    {
      title: "Machine Learning Basics",
      tags: ["ML", "Supervised Learning", "Regression"],
      description: "An introductory lecture on supervised learning.",
      total_time: 60,
      time_uploaded: Date.now() - 3600000,
    },
    {
      title: "Machine Learning Basics",
      tags: ["ML", "Supervised Learning", "Regression"],
      description: "An introductory lecture on supervised learning. adofjoajsdfasdfasjdfo;ajsdof;jasojdfoasjdfo;ajsdofijaosjdf;aosdfdafdsafsadfasdfasdfsda",
      total_time: 60,
      time_uploaded: Date.now() - 3600000,
    },
    {
      title: "Machine Learning Basics",
      tags: ["ML", "Supervised Learning", "Regression"],
      description: "An introductory lecture on supervised learning.",
      total_time: 60,
      time_uploaded: Date.now() - 3600000,
    },
    {
      title: "Machine Learning Basics",
      tags: ["ML", "Supervised Learning", "Regression"],
      description: "An introductory lecture on supervised learning.",
      total_time: 60,
      time_uploaded: Date.now() - 3600000,
    },
  ];

  return (
    <div>
      <Box sx={{ display: "flex", paddingTop: "40px" }}>
        <Sidebar />
        <div className="records-container">
          <div style={{ marginBottom: "20px" }}>
            <h1>My Records</h1>
          </div>
          <div className="all-records">
            {all_records.map((record, index) => (
              <Record
                key={index} // Use a unique key for each child in a list
                title={record.title}
                tags={record.tags}
                description={record.description}
                total_time={record.total_time}
                time_uploaded={record.time_uploaded}
              />
            ))}
          </div>
        </div>
      </Box>
    </div>
  );
};

export default MyRecords;
