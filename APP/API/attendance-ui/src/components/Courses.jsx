import React, { useState, useEffect } from 'react';
import axios from 'axios';

const Courses = ({ courseName }) => {
  const [courseData, setCourseData] = useState(null);

  useEffect(() => {
    // Fetch course data
    const fetchCourseData = async () => {
      try {
        const response = await axios.get(`/api/courses/${courseName}`);
        setCourseData(response.data);
      } catch (error) {
        console.error('Error fetching course data:', error);
      }
    };

    fetchCourseData();
  }, [courseName]);

  return (
    <div className="mt-4">
      <h2 className="text-2xl font-bold mb-4">Course Details</h2>
      {courseData ? (
        <div>
          <p>Course Name: {courseData.name}</p>
          {/* Add more course details as needed */}
        </div>
      ) : (
        <p>Loading course data...</p>
      )}
    </div>
  );
};

export default Courses;