"""
MaintenAI— Unit Tests for Configuration Module
"""


class TestSettings:
    """Test application settings loading and validation."""

    def test_default_settings_load(self):
        """Settings should load with defaults without any env vars."""
        from factorypulse.config import Settings

        settings = Settings()
        assert settings.app_name == "FactoryPulse"
        assert settings.app_env == "development"
        assert settings.debug is True

    def test_database_urls(self):
        """Database URL properties should construct valid connection strings."""
        from factorypulse.config import DatabaseSettings

        db = DatabaseSettings(
            host="testhost",
            port=5433,
            db="testdb",
            user="testuser",
            password="testpass",
        )
        assert "testhost:5433/testdb" in db.async_url
        assert "asyncpg" in db.async_url
        assert "psycopg2" in db.sync_url

    def test_redis_url(self):
        """Redis URL property should include password and port."""
        from factorypulse.config import RedisSettings

        redis = RedisSettings(host="redis-host", port=6380, password="secret", db=1)
        assert "redis-host:6380/1" in redis.url
        assert "secret" in redis.url

    def test_cors_origin_list(self):
        """CORS origins should be parsed from comma-separated string."""
        from factorypulse.config import Settings

        settings = Settings(cors_origins="http://a.com, http://b.com")
        assert settings.cors_origin_list == ["http://a.com", "http://b.com"]

    def test_alert_settings_defaults(self):
        """Alert thresholds should have sensible defaults."""
        from factorypulse.config import AlertSettings

        alerts = AlertSettings()
        assert alerts.anomaly_threshold == 0.85
        assert alerts.rul_threshold == 15
        assert alerts.failure_probability_threshold == 0.70
        assert alerts.hysteresis_cycles == 3
