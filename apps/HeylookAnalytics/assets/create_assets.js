// Simple script to create placeholder assets
const fs = require('fs');
const path = require('path');

// Create a simple 1x1 pixel PNG
const simplePNG = Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==', 'base64');

const assets = ['icon.png', 'splash.png', 'adaptive-icon.png', 'favicon.png'];

assets.forEach(asset => {
  fs.writeFileSync(path.join(__dirname, asset), simplePNG);
});

console.log('Created placeholder assets');