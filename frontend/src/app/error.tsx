"use client";
import React from 'react';

interface ErrorProps {
    message: string;
}

const ErrorLayout: React.FC<ErrorProps> = ({ message }) => {
    return (
        <div>
            <h1>Error</h1>
            <p>{message}</p>
        </div>
    );
};

export default ErrorLayout;
