"""Tests for plugin discovery and registration."""
import pytest


class TestPluginSystem:
    def test_plugin_registry_has_plugins(self):
        """After import, some plugins should be registered."""
        from yggdrasil.plugins.base import PluginRegistry
        import yggdrasil.plugins  # Trigger auto-discovery
        
        plugins = PluginRegistry.list_plugins()
        assert len(plugins) > 0
    
    def test_image_plugin_registered(self):
        """Image plugin should be auto-discovered."""
        from yggdrasil.plugins.base import PluginRegistry
        import yggdrasil.plugins
        
        plugins = PluginRegistry.list_names()
        assert "image" in plugins
    
    def test_plugin_has_ui_schema(self):
        """Each plugin should have a get_ui_schema method."""
        from yggdrasil.plugins.base import PluginRegistry
        import yggdrasil.plugins
        
        for name, cls in PluginRegistry.list_plugins().items():
            schema = cls.get_ui_schema()
            assert "inputs" in schema
            assert "outputs" in schema
    
    def test_plugin_has_available_configs(self):
        """Plugins should provide available configs."""
        from yggdrasil.plugins.base import PluginRegistry
        import yggdrasil.plugins
        
        for name, cls in PluginRegistry.list_plugins().items():
            configs = cls.get_available_configs()
            assert isinstance(configs, dict)
            assert len(configs) > 0
    
    def test_image_plugin_presets(self):
        """Image plugin should have SD15, SDXL, SD3, Flux presets."""
        from yggdrasil.plugins.base import PluginRegistry
        import yggdrasil.plugins
        
        image = PluginRegistry.get("image")
        configs = image.get_available_configs()
        
        assert "sd15" in configs
        assert "sdxl" in configs
        assert "sd3" in configs
        assert "flux" in configs


class TestPluginList:
    def test_list_plugins_function(self):
        """The convenience list_plugins function should work."""
        from yggdrasil.plugins import list_plugins
        
        plugins = list_plugins()
        assert isinstance(plugins, dict)
    
    def test_get_plugin_function(self):
        """The convenience get_plugin function should work."""
        from yggdrasil.plugins import get_plugin
        
        plugin = get_plugin("image")
        assert plugin.name == "image"
