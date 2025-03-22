import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { 
  Card, 
  CardContent, 
  Typography, 
  CircularProgress, 
  Alert,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper
} from '@mui/material';

const Attendance = ({ studentId }) => {
  const [attendance, setAttendance] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchAttendance = async () => {
      try {
        const response = await axios.get(`http://localhost:8000/attendance/${studentId}`);
        setAttendance(response.data.attendance_records);
      } catch (error) {
        setError('Failed to fetch attendance data');
        console.error('Error:', error);
      } finally {
        setLoading(false);
      }
    };

    if (studentId) {
      fetchAttendance();
    }
  }, [studentId]);

  if (loading) return (
    <Card sx={{ minWidth: 275, mt: 2, mb: 2, display: 'flex', justifyContent: 'center', p: 3 }}>
      <CircularProgress />
    </Card>
  );

  if (error) return (
    <Alert severity="error" sx={{ mt: 2, mb: 2 }}>
      {error}
    </Alert>
  );

  return (
    <Card sx={{ minWidth: 275, mt: 2, mb: 2 }}>
      <CardContent>
        <Typography variant="h5" component="div" gutterBottom>
          Attendance Record
        </Typography>
        
        <TableContainer component={Paper}>
          <Table sx={{ minWidth: 650 }} aria-label="attendance table">
            <TableHead>
              <TableRow>
                <TableCell>Course Name</TableCell>
                <TableCell>Date</TableCell>
                <TableCell>Morning Status</TableCell>
                <TableCell>Evening Status</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {attendance && attendance.map((record, index) => (
                <TableRow key={index}>
                  <TableCell>{record.course_name}</TableCell>
                  <TableCell>{new Date(record.date).toLocaleDateString()}</TableCell>
                  <TableCell sx={{
                    color: record.morning_status === 'Present' ? 'green' : 'red'
                  }}>
                    {record.morning_status}
                  </TableCell>
                  <TableCell sx={{
                    color: record.evening_status === 'Present' ? 'green' : 'red'
                  }}>
                    {record.evening_status}
                  </TableCell>
                </TableRow>
              ))}
              {(!attendance || attendance.length === 0) && (
                <TableRow>
                  <TableCell colSpan={4} align="center">No attendance records found</TableCell>
                </TableRow>
              )}
            </TableBody>
          </Table>
        </TableContainer>
      </CardContent>
    </Card>
  );
};

export default Attendance;