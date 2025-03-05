import React, { useState, useEffect, useRef } from 'react';
import { Input, Button, List, Avatar } from 'antd';

const { TextArea } = Input;

const App = () => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const messagesEndRef = useRef(null);

  const sendMessage = async () => {
    if (!input.trim()) return;
    const userMessage = { content: input, sender: 'user' };
    setMessages([...messages, userMessage]);
    setInput('');

    try {
      const response = await fetch('/send-message', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ message: input }),
      });

      const data = await response.json();
      const botMessage = {
        content: data.message,
        sender: 'bot',
        imageUri: data.imageUri,  // Add imageUri from the response if available
      };
      setMessages(messages => [...messages, botMessage]);
    } catch (error) {
      console.error('Error sending message:', error);
    }
  };

  // Scroll to the bottom of the chat when a new message is added
  useEffect(() => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages]);

  return (
    <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100vh' }}>
      <div style={{ width: '70%', maxWidth: '800px', height: '75%', border: '1px solid #f0f0f0', borderRadius: '8px', display: 'flex', flexDirection: 'column', position: 'relative' }}>
        <div style={{ flex: '1', padding: '20px', overflowY: 'auto' }}>
          <List
            dataSource={messages}
            renderItem={item => (
              <List.Item style={{ borderBottom: 'none' }}>
                <List.Item.Meta
                  avatar={<Avatar style={{ backgroundColor: item.sender === 'user' ? '#f56a00' : '#87d068' }} />}
                  title={item.sender === 'user' ? 'You' : 'Chatbot'}
                  description={
                    <div>
                      <p>{item.content}</p>
                      {item.imageUri && <img src={item.imageUri} alt="product" style={{ maxWidth: '100px', maxHeight: '100px' }} />}
                    </div>
                  }
                />
              </List.Item>
            )}
            itemLayout="horizontal"
            split={false}
          />
          <div ref={messagesEndRef} />
        </div>
        <div style={{ borderTop: '1px solid #f0f0f0', padding: '10px 20px', background: 'white', boxSizing: 'border-box' }}>
          <TextArea
            rows={2}
            value={input}
            onChange={e => setInput(e.target.value)}
            onPressEnter={sendMessage}
            style={{ marginRight: '10px', resize: 'none' }} // Prevent TextArea from being resized
          />
          <Button type="primary" onClick={sendMessage}>
            Send
          </Button>
        </div>
      </div>
    </div>
  );
};

export default App;
