import React, { useState, useEffect } from "react";
import { Container, Box, Tabs, Tab, Typography } from '@mui/material';
import Navbar from "./components/Navbar";
import Login from "./components/Login";
import Register from "./components/Register";
import Attendance from "./components/Attendance";
import Courses from "./components/Courses";
import SignatureValidation from "./components/SignatureValidation";
import AttendanceTable from './components/AttendanceTable';
import SignatureTable from './components/SignatureTable';
import Dashboard from './components/Dashboard';

function App() {
  const [tab, setTab] = useState(0);
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [userData, setUserData] = useState(null);

  useEffect(() => {
    const storedUser = localStorage.getItem('user');
    if (storedUser) {
      try {
        const user = JSON.parse(storedUser);
        setUserData(user);
        setIsAuthenticated(true);
      } catch (error) {
        console.error('Error parsing stored user data:', error);
        localStorage.removeItem('user');
      }
    }
  }, []);

  const handleLoginSuccess = (data) => {
    const userInfo = {
      id: data.user_id,
      email: data.email,
      firstName: data.first_name,
      lastName: data.last_name,
      attendanceRecords: data.attendance_records,
      signatureRecords: data.signature_records
    };
    setUserData(userInfo);
    setIsAuthenticated(true);
    localStorage.setItem('user', JSON.stringify(userInfo));
  };

  const handleLogout = () => {
    setIsAuthenticated(false);
    setUserData(null);
    localStorage.removeItem('user');
    localStorage.removeItem('token');
  };

  const handleTabChange = (event, newValue) => {
    setTab(newValue);
  };

  return (
    <Box sx={{ minHeight: '100vh', bgcolor: '#f5f5f5' }}>
      <Navbar isAuthenticated={isAuthenticated} onLogout={handleLogout} />
      <Container maxWidth="lg" sx={{ py: 4 }}>
        {!isAuthenticated ? (
          <Box sx={{ width: '100%' }}>
            <Box sx={{ borderBottom: 1, borderColor: 'divider', mb: 3 }}>
              <Tabs value={tab} onChange={handleTabChange} centered>
                <Tab label="Login" />
                <Tab label="Register" />
              </Tabs>
            </Box>
            <Box sx={{ p: 3 }}>
              {tab === 0 && <Login onLoginSuccess={handleLoginSuccess} />}
              {tab === 1 && <Register onRegisterSuccess={() => setTab(0)} />}
            </Box>
          </Box>
        ) : (
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
            {userData && (
              <>
                <Typography variant="h4" gutterBottom>
                  Welcome, {userData.firstName} {userData.lastName}
                </Typography>
                
                <Dashboard 
                  attendanceRecords={userData.attendanceRecords}
                  signatureRecords={userData.signatureRecords}
                />
                
                <Attendance studentId={userData.id} />
                <SignatureValidation userId={userData.id} />
              </>
            )}
          </Box>
        )}
      </Container>
    </Box>
  );
}

export default App;
