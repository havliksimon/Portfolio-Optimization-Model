"""
Portfolio Optimizer - Authentication Models
============================================

User management, authentication, and authorization models.
Implements admin approval workflow for registration.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.dialects.postgresql import JSONB
from werkzeug.security import generate_password_hash, check_password_hash
import secrets
import string

from models.database import db


class User(db.Model):
    """
    User entity with role-based access control.
    
    Registration workflow:
    1. User registers (status='pending')
    2. Admin receives email notification
    3. Admin approves/rejects registration
    4. If approved, user can login
    """
    
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(255), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    
    # Profile
    first_name = db.Column(db.String(100))
    last_name = db.Column(db.String(100))
    
    # Role and status
    role = db.Column(db.String(20), default='user')  # admin, user, viewer
    status = db.Column(db.String(20), default='pending')  # pending, active, suspended, rejected
    
    # Email verification
    email_verified = db.Column(db.Boolean, default=False)
    email_verified_at = db.Column(db.DateTime)
    
    # Registration approval
    approved_by = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)
    approved_at = db.Column(db.DateTime)
    rejection_reason = db.Column(db.Text)
    
    # Security
    failed_login_attempts = db.Column(db.Integer, default=0)
    locked_until = db.Column(db.DateTime)
    last_login = db.Column(db.DateTime)
    last_login_ip = db.Column(db.String(45))  # IPv6 compatible
    
    # Preferences
    preferences = db.Column(JSONB().with_variant(db.JSON, "sqlite"), default=dict)
    
    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    portfolios = db.relationship('Portfolio', backref='owner', lazy='dynamic',
                                foreign_keys='Portfolio.user_id')
    saved_scenarios = db.relationship('SavedScenario', backref='user', lazy='dynamic')
    approval_record = db.relationship('User', remote_side=[id], 
                                     foreign_keys=[approved_by])
    
    def __repr__(self) -> str:
        return f"<User {self.email}: {self.role} ({self.status})>"
    
    def set_password(self, password: str):
        """Hash and set user password."""
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password: str) -> bool:
        """Verify password against hash."""
        return check_password_hash(self.password_hash, password)
    
    def is_active(self) -> bool:
        """Check if user account is active and not locked."""
        if self.status != 'active':
            return False
        if self.locked_until and self.locked_until > datetime.utcnow():
            return False
        return True
    
    def is_admin(self) -> bool:
        """Check if user has admin role."""
        return self.role == 'admin'
    
    def approve(self, admin_user_id: int):
        """Approve pending registration."""
        self.status = 'active'
        self.approved_by = admin_user_id
        self.approved_at = datetime.utcnow()
    
    def reject(self, admin_user_id: int, reason: str = None):
        """Reject pending registration."""
        self.status = 'rejected'
        self.approved_by = admin_user_id
        self.approved_at = datetime.utcnow()
        self.rejection_reason = reason
    
    def record_login(self, ip_address: str = None):
        """Record successful login."""
        self.last_login = datetime.utcnow()
        self.last_login_ip = ip_address
        self.failed_login_attempts = 0
        self.locked_until = None
    
    def record_failed_login(self):
        """Record failed login attempt and lock if necessary."""
        self.failed_login_attempts += 1
        if self.failed_login_attempts >= 5:
            self.locked_until = datetime.utcnow() + timedelta(minutes=30)
    
    def generate_reset_token(self) -> str:
        """Generate password reset token."""
        token = PasswordResetToken(
            user_id=self.id,
            token=secrets.token_urlsafe(32),
            expires_at=datetime.utcnow() + timedelta(hours=24)
        )
        db.session.add(token)
        db.session.commit()
        return token.token
    
    def to_dict(self, include_sensitive: bool = False) -> Dict[str, Any]:
        """Convert user to dictionary."""
        data = {
            'id': self.id,
            'email': self.email,
            'first_name': self.first_name,
            'last_name': self.last_name,
            'role': self.role,
            'status': self.status,
            'email_verified': self.email_verified,
            'created_at': self.created_at.isoformat(),
            'last_login': self.last_login.isoformat() if self.last_login else None,
        }
        
        if include_sensitive:
            data.update({
                'failed_login_attempts': self.failed_login_attempts,
                'locked_until': self.locked_until.isoformat() if self.locked_until else None,
            })
        
        return data


class PasswordResetToken(db.Model):
    """Password reset token with expiration."""
    
    __tablename__ = 'password_reset_tokens'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    token = db.Column(db.String(255), unique=True, nullable=False, index=True)
    
    expires_at = db.Column(db.DateTime, nullable=False)
    used_at = db.Column(db.DateTime)
    used = db.Column(db.Boolean, default=False)
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    user = db.relationship('User')
    
    def is_valid(self) -> bool:
        """Check if token is still valid."""
        return not self.used and self.expires_at > datetime.utcnow()
    
    def mark_used(self):
        """Mark token as used."""
        self.used = True
        self.used_at = datetime.utcnow()


class SavedScenario(db.Model):
    """
    Saved portfolio scenario for registered users.
    
    Stores portfolio configurations and analysis results
    for later retrieval and comparison.
    """
    
    __tablename__ = 'saved_scenarios'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    
    name = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text)
    
    # Portfolio configuration
    holdings = db.Column(JSONB().with_variant(db.JSON, "sqlite"), nullable=False)
    period = db.Column(db.String(10), default='2y')
    
    # Analysis results (cached)
    analysis_results = db.Column(JSONB().with_variant(db.JSON, "sqlite"), default=dict)
    
    # Tags for organization
    tags = db.Column(JSONB().with_variant(db.JSON, "sqlite"), default=list)
    
    # Sharing
    is_public = db.Column(db.Boolean, default=False)
    share_token = db.Column(db.String(255), unique=True, index=True)
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self) -> str:
        return f"<SavedScenario {self.name} by User {self.user_id}>"
    
    def generate_share_token(self):
        """Generate unique share token."""
        self.share_token = secrets.token_urlsafe(16)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'holdings': self.holdings,
            'period': self.period,
            'tags': self.tags,
            'is_public': self.is_public,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
        }


class UserActivity(db.Model):
    """Audit log for user activities."""
    
    __tablename__ = 'user_activities'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    
    action = db.Column(db.String(50), nullable=False)  # login, logout, save_scenario, run_analysis, etc.
    details = db.Column(JSONB().with_variant(db.JSON, "sqlite"), default=dict)
    ip_address = db.Column(db.String(45))
    user_agent = db.Column(db.String(500))
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    user = db.relationship('User')
    
    def __repr__(self) -> str:
        return f"<UserActivity {self.action} by User {self.user_id}>"


class GuestSession(db.Model):
    """
    Temporary storage for guest users (non-registered).
    
    Data stored in-memory/SQLite only, not in production PostgreSQL.
    """
    
    __tablename__ = 'guest_sessions'
    
    id = db.Column(db.Integer, primary_key=True)
    session_token = db.Column(db.String(255), unique=True, nullable=False, index=True)
    
    # Temporary data storage
    temp_data = db.Column(JSONB().with_variant(db.JSON, "sqlite"), default=dict)
    
    # Expiration
    expires_at = db.Column(db.DateTime, nullable=False)
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_accessed = db.Column(db.DateTime, default=datetime.utcnow)
    
    def is_valid(self) -> bool:
        """Check if session is still valid."""
        return self.expires_at > datetime.utcnow()
    
    def touch(self):
        """Update last accessed timestamp."""
        self.last_accessed = datetime.utcnow()
