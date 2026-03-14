from yggdrasill.task_nodes.roles import Role, role_from_block_type, ALL_ROLES


class TestRoleEnum:
    def test_all_seven_roles(self):
        assert len(ALL_ROLES) == 7
        names = {r.value for r in ALL_ROLES}
        assert names == {
            "backbone", "injector", "conjector",
            "inner_module", "outer_module",
            "helper", "converter",
        }


class TestRoleFromBlockType:
    def test_backbone(self):
        assert role_from_block_type("backbone/identity") == Role.BACKBONE

    def test_injector(self):
        assert role_from_block_type("injector/clip") == Role.INJECTOR

    def test_converter(self):
        assert role_from_block_type("converter/vae_enc") == Role.CONVERTER

    def test_unknown_prefix(self):
        assert role_from_block_type("unknown/foo") is None

    def test_no_slash(self):
        assert role_from_block_type("backbone") == Role.BACKBONE

    def test_case_insensitive(self):
        assert role_from_block_type("BACKBONE/Identity") == Role.BACKBONE
