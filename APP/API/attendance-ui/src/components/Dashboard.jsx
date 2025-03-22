import React, { useEffect, useState } from 'react';
import { 
  Card, 
  CardContent, 
  Typography, 
  Grid,
  Alert,
  Box 
} from '@mui/material';
import {
  LineChart,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell
} from 'recharts';

const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042'];

const Dashboard = ({ attendanceRecords: initialAttendance, signatureRecords: initialSignature, userId }) => {
  const [attendanceRecords, setAttendanceRecords] = useState(initialAttendance);
  const [signatureRecords, setSignatureRecords] = useState(initialSignature);
  const [ws, setWs] = useState(null);

  useEffect(() => {
    const websocket = new WebSocket(`ws://localhost:8000/ws/${userId}`);
    
    websocket.onmessage = (event) => {
      const data = JSON.parse(event.data);
      setAttendanceRecords(prevRecords => {
        // Merge existing and new records, removing duplicates by date
        const mergedRecords = [...prevRecords];
        data.attendance_records.forEach(newRecord => {
          const existingIndex = mergedRecords.findIndex(
            record => record.date === newRecord.date
          );
          if (existingIndex === -1) {
            mergedRecords.push(newRecord);
          }
        });
        return mergedRecords;
      });

      setSignatureRecords(prevRecords => {
        // Merge existing and new signature records
        const mergedRecords = [...prevRecords];
        data.signature_records.forEach(newRecord => {
          const existingIndex = mergedRecords.findIndex(
            record => record.date === newRecord.date
          );
          if (existingIndex === -1) {
            mergedRecords.push(newRecord);
          }
        });
        return mergedRecords;
      });
    };

    websocket.onerror = (error) => {
      console.error('WebSocket error:', error);
    };

    setWs(websocket);

    return () => {
      websocket.close();
    };
  }, [userId]);

  // Process attendance data
  const attendanceStats = attendanceRecords?.reduce((acc, record) => {
    if (!acc.morning) acc.morning = { Present: 0, Absent: 0 };
    if (!acc.evening) acc.evening = { Present: 0, Absent: 0 };
    
    acc.morning[record.morning_status] = (acc.morning[record.morning_status] || 0) + 1;
    acc.evening[record.evening_status] = (acc.evening[record.evening_status] || 0) + 1;
    
    return acc;
  }, { morning: { Present: 0, Absent: 0 }, evening: { Present: 0, Absent: 0 } });

  const attendancePieData = [
    { name: 'Morning Present', value: attendanceStats?.morning?.Present || 0 },
    { name: 'Morning Absent', value: attendanceStats?.morning?.Absent || 0 },
    { name: 'Evening Present', value: attendanceStats?.evening?.Present || 0 },
    { name: 'Evening Absent', value: attendanceStats?.evening?.Absent || 0 }
  ].filter(item => item.value > 0);

  // Process signature data
  const signatureData = signatureRecords?.map(record => ({
    date: new Date(record.date).toLocaleDateString(),
    morningConfidence: parseFloat(record.Morning_Confidence || 0),
    eveningConfidence: parseFloat(record.Evening_Confidence || 0)
  })).sort((a, b) => new Date(a.date) - new Date(b.date)) || [];

  // Calculate attendance percentage
  const calculateAttendance = () => {
    if (!attendanceRecords?.length) return 0;
    const totalSessions = attendanceRecords.length * 2; // Total possible sessions (morning + evening)
    const presentSessions = attendanceRecords.reduce((acc, record) => {
      const morningCount = record.morning_status === "Present" ? 1 : 0;
      const eveningCount = record.evening_status === "Present" ? 1 : 0;
      return acc + morningCount + eveningCount;
    }, 0);
    
    // For your data: 3 records × 2 sessions = 6 total sessions
    // Present sessions: 3 morning (Present) + 0 evening (all Absent) = 3
    // 3/6 × 100 = 50%
    return (presentSessions / totalSessions) * 100;
  };

  // Calculate forged signature percentage
  const calculateForgedSignatures = () => {
    if (!signatureRecords?.length) return 0;
    const totalSignatures = signatureRecords.length * 2;
    const forgedCount = signatureRecords.reduce((acc, record) => {
      return acc + 
        (record.Morning_Prediction === "forged" ? 1 : 0) +
        (record.Evening_Prediction === "forged" ? 1 : 0);
    }, 0);
    return (forgedCount / totalSignatures) * 100;
  };

  const attendancePercentage = calculateAttendance();
  const forgedPercentage = calculateForgedSignatures();

  return (
    <Grid container spacing={3}>
      {/* Warning Alerts */}
      <Grid item xs={12}>
        <Box mb={3}>
          {attendancePercentage < 70 && (
            <Alert severity="warning" sx={{ mb: 2 }}>
              Your attendance is {attendancePercentage.toFixed(1)}%, which is below the required 70%
            </Alert>
          )}
          {forgedPercentage > 50 && (
            <Alert severity="error">
              Warning: {forgedPercentage.toFixed(1)}% of your signatures have been detected as potentially forged
            </Alert>
          )}
        </Box>
      </Grid>

      {/* Statistics Cards */}
      <Grid item xs={12} md={6}>
        <Card>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              Overall Attendance
            </Typography>
            <Typography 
              variant="h3" 
              color={attendancePercentage < 70 ? "error" : "success.main"}
            >
              {attendancePercentage.toFixed(1)}%
            </Typography>
          </CardContent>
        </Card>
      </Grid>

      <Grid item xs={12} md={6}>
        <Card>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              Signature Authenticity
            </Typography>
            <Typography 
              variant="h3" 
              color={forgedPercentage > 50 ? "error" : "success.main"}
            >
              {(100 - forgedPercentage).toFixed(1)}%
            </Typography>
            <Typography variant="body2" color="textSecondary">
              Genuine Signatures
            </Typography>
          </CardContent>
        </Card>
      </Grid>

      {/* Charts */}
      <Grid item xs={12} md={6}>
        <Card>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              Attendance Distribution
            </Typography>
            <ResponsiveContainer width="100%" height={300}>
              <PieChart>
                <Pie
                  data={attendancePieData}
                  dataKey="value"
                  nameKey="name"
                  cx="50%"
                  cy="50%"
                  outerRadius={80}
                  label={({ name, value, percent }) => 
                    `${name}: ${value} (${(percent * 100).toFixed(0)}%)`
                  }
                >
                  {attendancePieData.map((entry, index) => (
                    <Cell 
                      key={`cell-${index}`} 
                      fill={COLORS[index % COLORS.length]}
                    />
                  ))}
                </Pie>
                <Tooltip />
                <Legend verticalAlign="bottom" height={36} />
              </PieChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </Grid>

      <Grid item xs={12} md={6}>
        <Card>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              Signature Confidence Trends
            </Typography>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={signatureData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis 
                  dataKey="date"
                  angle={-45}
                  textAnchor="end"
                  height={60}
                  interval={0}
                />
                <YAxis 
                  domain={[0, 100]}
                  ticks={[0, 25, 50, 75, 100]}
                />
                <Tooltip 
                  formatter={(value) => [`${value}%`]}
                  labelFormatter={(label) => `Date: ${label}`}
                />
                <Legend />
                <Line 
                  type="monotone" 
                  dataKey="morningConfidence" 
                  name="Morning Confidence" 
                  stroke="#8884d8"
                  strokeWidth={2}
                  dot={{ r: 4 }}
                />
                <Line 
                  type="monotone" 
                  dataKey="eveningConfidence" 
                  name="Evening Confidence" 
                  stroke="#82ca9d"
                  strokeWidth={2}
                  dot={{ r: 4 }}
                />
              </LineChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </Grid>
    </Grid>
  );
};

export default Dashboard;

