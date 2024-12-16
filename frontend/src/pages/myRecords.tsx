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
      title: "L6-Classsification-917_1-3.pdf",
      tags: ["Computer Science", "Machine Learning"],
      description: "A lecture anout generative and discriminative classification",
      total_time: 8,
      time_uploaded: Date.now(),
    },
    {
      title: "cs168-fa24-lec07.pdf",
      tags: ["Computer Science", "Internet"],
      description: "A lecture about Distance Vector Protocol",
      total_time: 40,
      time_uploaded: Date.now(),
    },
    {
      title: "Agent_and_LLM_Definitions.pdf",
      tags: ["Computer Science", "Artificial Intelligence"],
      description: "A lecture about Artificial Intelligence and large language model agents",
      total_time: 10,
      time_uploaded: Date.now(),
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
