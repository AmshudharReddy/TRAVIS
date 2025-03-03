import React, { memo } from 'react';

const About = memo(() => {
    return (
        <>
            <div style={{ fontSize: "24px", margin: "30px", marginBottom: "10px", color: "#555" }}>
                Our project is based on:
            </div>
            <p style={{ fontSize: "40px", margin: "20px", fontWeight: "bold" }}>
                Transformer system helping Visually Impaired Service Agents
            </p>

        </>
    );
});

export default About;
