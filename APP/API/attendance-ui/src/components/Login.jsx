import React, { useState } from 'react';
import axios from 'axios';
import {
  Card,
  CardContent,
  TextField,
  Button,
  Typography,
  Alert,
  Box,
  CircularProgress
} from '@mui/material';

const Login = ({ onLoginSuccess }) => {
  const [credentials, setCredentials] = useState({
    username: '',
    password: ''
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleChange = (e) => {
    setCredentials({
      ...credentials,
      [e.target.name]: e.target.value
    });
  };

  const handleLogin = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    try {
      const response = await axios.post('http://localhost:8000/login', {
        email: credentials.username,
        password: credentials.password
      });
      
      if (response && response.data) {
        localStorage.setItem('token', response.data.message);
        localStorage.setItem('user', JSON.stringify({
          id: response.data.user_id,
          email: response.data.email,
          firstName: response.data.first_name,
          lastName: response.data.last_name,
          attendanceRecords: response.data.attendance_records,
          signatureRecords: response.data.signature_records
        }));
        onLoginSuccess(response.data); // Pass the data to parent component
      }
    } catch (err) {
      console.error('Login error:', err);
      const errorMessage = err.response?.data?.detail;
      setError(Array.isArray(errorMessage) ? errorMessage[0]?.msg : errorMessage || 'Login failed. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <Card sx={{ width: '100%', maxWidth: 400, mx: 'auto' }}>
      <CardContent>
        <Typography variant="h5" component="h2" gutterBottom>
          Login
        </Typography>
        
        {error && (
          <Alert severity="error" sx={{ mb: 2 }}>
            {typeof error === 'string' ? error : 'Login failed. Please try again.'}
          </Alert>
        )}

        <Box 
          component="form" 
          onSubmit={handleLogin} 
          sx={{ 
            display: 'flex', 
            flexDirection: 'column', 
            gap: 2 
          }}
        >
          <TextField
            required
            label="Email"  // Changed from Username to Email
            name="username"
            type="email"   // Added email type
            value={credentials.username}
            onChange={handleChange}
            fullWidth
          />
          
          <TextField
            required
            label="Password"
            name="password"
            type="password"
            value={credentials.password}
            onChange={handleChange}
            fullWidth
          />

          <Button 
            type="submit" 
            variant="contained" 
            color="primary"
            disabled={loading}
            sx={{ mt: 1 }}
            fullWidth
          >
            {loading ? (
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <CircularProgress size={20} color="inherit" />
                <span>Logging in...</span>
              </Box>
            ) : (
              'Login'
            )}
          </Button>
        </Box>
      </CardContent>
    </Card>
  );
};

export default Login;