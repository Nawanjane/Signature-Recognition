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

const SignatureTable = ({ signatureRecords }) => {
  if (!signatureRecords || signatureRecords.length === 0) {
    return (
      <Typography color="text.secondary" align="center">
        No signature validation records found
      </Typography>
    );
  }

  return (
    <TableContainer component={Paper}>
      <Table>
        <TableHead>
          <TableRow>
            <TableCell>Name</TableCell>
            <TableCell>Morning Prediction</TableCell>
            <TableCell>Morning Confidence</TableCell>
            <TableCell>Evening Prediction</TableCell>
            <TableCell>Evening Confidence</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {signatureRecords.map((record, index) => (
            <TableRow key={index}>
              <TableCell>{record.Name}</TableCell>
              <TableCell>{record.Morning_Prediction}</TableCell>
              <TableCell>{`${(record.Morning_Confidence * 100).toFixed(2)}%`}</TableCell>
              <TableCell>{record.Evening_Prediction}</TableCell>
              <TableCell>{`${(record.Evening_Confidence * 100).toFixed(2)}%`}</TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </TableContainer>
  );
};

export default SignatureTable;