const express = require('express');
const app = express();
const port = 5000;
const cors =require('cors');
const connectToMongo = require('./db');


connectToMongo();
app.use(cors())


app.use(express.json());


// Available Routes
app.use('/api/auth', require('./routes/auth'));
app.use('/api/query', require('./routes/query'));


app.get('/', (req, res)=> {
    res.send("Riya ekkada?");
})


app.listen(port, '0.0.0.0', ()=>{
    console.log(`TRAVIS backend is listening on port ${port}`);
})