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

const SignatureValidation = ({ userId }) => {
  const [validation, setValidation] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchSignatureValidation = async () => {
      try {
        const response = await axios.get(`http://localhost:8000/signature_validation/${userId}`);
        setValidation(response.data.signature_records);
      } catch (error) {
        setError('Failed to fetch signature validation data');
        console.error('Error:', error);
      } finally {
        setLoading(false);
      }
    };

    if (userId) {
      fetchSignatureValidation();
    }
  }, [userId]);

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
          Signature Validation Records
        </Typography>

        <TableContainer component={Paper}>
          <Table sx={{ minWidth: 650 }} aria-label="signature validation table">
            <TableHead>
              <TableRow>
                <TableCell>Date</TableCell>
                <TableCell>Name</TableCell>
                <TableCell>Morning Prediction</TableCell>
                <TableCell>Morning Confidence</TableCell>
                <TableCell>Evening Prediction</TableCell>
                <TableCell>Evening Confidence</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {validation && validation.map((record, index) => (
                <TableRow key={index}>
                  <TableCell>{record.date}</TableCell>
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
      </CardContent>
    </Card>
  );
};

export default SignatureValidation;