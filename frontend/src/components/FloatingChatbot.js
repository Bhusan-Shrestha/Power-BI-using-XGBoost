import React, { useEffect, useMemo, useRef, useState } from 'react';
import {
  Avatar,
  Badge,
  Button,
  Card,
  Input,
  Tag,
  Typography,
} from 'antd';
import {
  BulbOutlined,
  CloseOutlined,
  MessageOutlined,
  RobotOutlined,
  SendOutlined,
} from '@ant-design/icons';
import axios from '../api/axios';
import { useAuthStore } from '../store/authStore';

const { Text } = Typography;

function ChatBubble({ role, content, time, userName }) {
  const isUser = role === 'user';
  return (
    <div
      style={{
        display: 'flex',
        justifyContent: isUser ? 'flex-end' : 'flex-start',
        gap: 8,
        marginBottom: 10,
      }}
    >
      {!isUser && (
        <Avatar size={28} icon={<RobotOutlined />} style={{ background: '#0f6fff' }} />
      )}
      <div
        style={{
          maxWidth: '80%',
          whiteSpace: 'pre-wrap',
          wordBreak: 'break-word',
          fontSize: 13,
          lineHeight: 1.5,
          padding: '10px 12px',
          borderRadius: isUser ? '12px 12px 4px 12px' : '12px 12px 12px 4px',
          background: isUser ? '#0f6fff' : '#f4f7ff',
          color: isUser ? '#fff' : '#14213d',
          boxShadow: '0 1px 4px rgba(0,0,0,0.08)',
        }}
      >
        {content}
        <div
          style={{
            marginTop: 4,
            textAlign: 'right',
            fontSize: 10,
            opacity: isUser ? 0.75 : 0.55,
          }}
        >
          {time}
        </div>
      </div>
      {isUser && (
        <Avatar size={28} style={{ background: '#14b8a6', fontWeight: 700 }}>
          {userName?.[0]?.toUpperCase() || 'U'}
        </Avatar>
      )}
    </div>
  );
}

export default function FloatingChatbot() {
  const { user } = useAuthStore();
  const [open, setOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const [input, setInput] = useState('');
  const [suggestions, setSuggestions] = useState([]);
  const [messages, setMessages] = useState([
    {
      role: 'assistant',
      content:
        'Hi! I am your BI assistant. Ask from suggested questions and I will answer based on current business data.',
      time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    },
  ]);

  const scrollerRef = useRef(null);

  useEffect(() => {
    const loadSuggestions = async () => {
      try {
        const res = await axios.get('/api/chatbot/suggestions');
        setSuggestions(res.data?.suggestions || []);
      } catch (e) {
        setSuggestions([]);
      }
    };
    loadSuggestions();
  }, []);

  useEffect(() => {
    if (!open) return;
    scrollerRef.current?.scrollTo({ top: scrollerRef.current.scrollHeight, behavior: 'smooth' });
  }, [messages, loading, open]);

  const now = () => new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

  const questionSet = useMemo(() => new Set(suggestions.map(q => q.toLowerCase())), [suggestions]);

  const sendMessage = async (text) => {
    const msg = (text || input).trim();
    if (!msg) return;

    setInput('');
    setMessages(prev => [...prev, { role: 'user', content: msg, time: now() }]);

    if (suggestions.length > 0 && !questionSet.has(msg.toLowerCase())) {
      setMessages(prev => [
        ...prev,
        {
          role: 'assistant',
          content: 'Please use one of the predefined questions below so I can answer accurately from your data.',
          time: now(),
        },
      ]);
      return;
    }

    setLoading(true);
    try {
      const res = await axios.post('/api/chatbot/chat', {
        message: msg,
        session_id: `widget-${user?.id || 'guest'}`,
      });

      const reply = typeof res.data?.response === 'string'
        ? res.data.response
        : JSON.stringify(res.data?.response || 'No response', null, 2);

      setMessages(prev => [...prev, { role: 'assistant', content: reply, time: now() }]);
    } catch (e) {
      setMessages(prev => [
        ...prev,
        {
          role: 'assistant',
          content: 'I could not fetch the answer right now. Please try again.',
          time: now(),
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      {open && (
        <div
          style={{
            position: 'fixed',
            right: 24,
            bottom: 94,
            zIndex: 1200,
            width: 370,
            maxWidth: 'calc(100vw - 20px)',
          }}
        >
          <Card
            bodyStyle={{ padding: 0 }}
            style={{ borderRadius: 16, overflow: 'hidden', boxShadow: '0 16px 40px rgba(0,0,0,0.2)' }}
            title={
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  <Avatar icon={<RobotOutlined />} style={{ background: '#0f6fff' }} />
                  <div>
                    <div style={{ fontWeight: 700, lineHeight: 1.2 }}>AI Assistant</div>
                    <Text type="secondary" style={{ fontSize: 11 }}>
                      Data-backed predefined answers
                    </Text>
                  </div>
                </div>
                <Button size="small" type="text" icon={<CloseOutlined />} onClick={() => setOpen(false)} />
              </div>
            }
          >
            <div
              ref={scrollerRef}
              style={{
                height: 320,
                overflowY: 'auto',
                background: 'linear-gradient(180deg, #f8fbff 0%, #ffffff 100%)',
                padding: 12,
              }}
            >
              {messages.map((m, idx) => (
                <ChatBubble key={idx} role={m.role} content={m.content} time={m.time} userName={user?.full_name} />
              ))}
              {loading && <Tag color="blue">AI is typing...</Tag>}
            </div>

            <div style={{ padding: 12, borderTop: '1px solid #eef2f7' }}>
              <div style={{ display: 'flex', gap: 6, marginBottom: 10, flexWrap: 'wrap', maxHeight: 120, overflowY: 'auto' }}>
                {suggestions.map((q) => (
                  <Button
                    key={q}
                    size="small"
                    icon={<BulbOutlined />}
                    onClick={() => sendMessage(q)}
                    disabled={loading}
                  >
                    {q.length > 26 ? `${q.slice(0, 26)}...` : q}
                  </Button>
                ))}
              </div>
              <div style={{ display: 'flex', gap: 8 }}>
                <Input
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onPressEnter={() => sendMessage()}
                  placeholder="Pick or type predefined question"
                  disabled={loading}
                />
                <Button type="primary" icon={<SendOutlined />} loading={loading} onClick={() => sendMessage()} />
              </div>
            </div>
          </Card>
        </div>
      )}

      <Badge dot offset={[-6, 6]}>
        <Button
          type="primary"
          shape="circle"
          size="large"
          icon={<MessageOutlined />}
          onClick={() => setOpen(v => !v)}
          style={{
            position: 'fixed',
            right: 24,
            bottom: 24,
            zIndex: 1200,
            width: 56,
            height: 56,
            boxShadow: '0 10px 24px rgba(15,111,255,0.45)',
            background: 'linear-gradient(135deg, #0f6fff 0%, #0a58ca 100%)',
            border: 'none',
          }}
        />
      </Badge>
    </>
  );
}
