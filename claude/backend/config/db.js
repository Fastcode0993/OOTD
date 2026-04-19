const mongoose = require('mongoose');

const connectDB = async () => {
  try {
    const conn = await mongoose.connect(process.env.MONGODB_URI, {
      serverSelectionTimeoutMS: 5000,
    });
    console.log(`MongoDB 연결됨: ${conn.connection.host}`);
  } catch (err) {
    console.error(`MongoDB 연결 실패: ${err.message}`);
    process.exit(1);
  }
};

module.exports = connectDB;
