import React from 'react';
import {
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Typography
} from '@mui/material';

const AttendanceTable = ({ attendanceRecords }) => {
  if (!attendanceRecords || attendanceRecords.length === 0) {
    return (
      <Typography color="text.secondary" align="center">
        No attendance records found
      </Typography>
    );
  }

  return (
    <TableContainer component={Paper}>
      <Table>
        <TableHead>
          <TableRow>
            <TableCell>Course Name</TableCell>
            <TableCell>Date</TableCell>
            <TableCell>Status</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {attendanceRecords.map((record, index) => (
            <TableRow key={index}>
              <TableCell>{record.course_name}</TableCell>
              <TableCell>{record.date}</TableCell>
              <TableCell>{record.attendance[0]?.status || 'N/A'}</TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </TableContainer>
  );
};

export default AttendanceTable;