import pytest

from tests.foundation.helpers import AddBlock, BlockWithSub, IdentityBlock


class TestBlockIdentity:
    def test_block_type(self):
        b = IdentityBlock()
        assert b.block_type == "test/identity"

    def test_block_id_auto(self):
        b = IdentityBlock()
        assert b.block_id  # non-empty

    def test_block_id_explicit(self):
        b = IdentityBlock(block_id="my_id")
        assert b.block_id == "my_id"

    def test_config_empty(self):
        b = IdentityBlock()
        assert b.config == {}

    def test_config_copy(self):
        cfg = {"foo": 1}
        b = IdentityBlock(config=cfg)
        assert b.config == {"foo": 1}
        b.config["bar"] = 2  # mutating the copy
        assert "bar" not in b._config  # original untouched


class TestBlockForward:
    def test_identity(self):
        b = IdentityBlock()
        assert b.forward({"x": 42}) == {"y": 42}

    def test_add(self):
        b = AddBlock(config={"offset": 10})
        assert b.forward({"a": 1, "b": 2}) == {"out": 13}

    def test_add_default_offset(self):
        b = AddBlock()
        assert b.forward({"a": 3, "b": 4}) == {"out": 7}

    def test_missing_key(self):
        b = AddBlock()
        with pytest.raises(KeyError):
            b.forward({"a": 1})


class TestBlockState:
    def test_identity_state_empty(self):
        b = IdentityBlock()
        assert b.state_dict() == {}

    def test_add_state_dict(self):
        b = AddBlock(config={"offset": 5})
        assert b.state_dict() == {"offset": 5}

    def test_load_state_dict(self):
        b = AddBlock(config={"offset": 0})
        b.load_state_dict({"offset": 99})
        assert b.offset == 99
        assert b.forward({"a": 0, "b": 0}) == {"out": 99}

    def test_load_state_strict_extra_key(self):
        b = AddBlock()
        with pytest.raises(KeyError):
            b.load_state_dict({"offset": 1, "extra": 2}, strict=True)

    def test_load_state_not_strict(self):
        b = AddBlock()
        b.load_state_dict({"offset": 7, "extra": 99}, strict=False)
        assert b.offset == 7


class TestBlockTrainFreeze:
    def test_default_training(self):
        b = IdentityBlock()
        assert b.training is True

    def test_eval(self):
        b = IdentityBlock()
        b.eval()
        assert b.training is False

    def test_train(self):
        b = IdentityBlock()
        b.eval()
        b.train()
        assert b.training is True

    def test_freeze_unfreeze(self):
        b = IdentityBlock()
        assert b.frozen is False
        b.freeze()
        assert b.frozen is True
        b.unfreeze()
        assert b.frozen is False


class TestBlockWithSubBlocks:
    def test_sub_state_dict(self):
        b = BlockWithSub(config={"child_offset": 3})
        assert b.state_dict() == {"child.offset": 3}

    def test_sub_load_state_dict(self):
        b = BlockWithSub(config={"child_offset": 0})
        b.load_state_dict({"child.offset": 42})
        assert b.child.offset == 42

    def test_sub_forward(self):
        b = BlockWithSub(config={"child_offset": 10})
        assert b.forward({"a": 1, "b": 2}) == {"out": 13}

    def test_sub_train_propagates(self):
        b = BlockWithSub()
        b.eval()
        assert b.child.training is False
        b.train()
        assert b.child.training is True


class TestBlockGetConfig:
    def test_get_config(self):
        b = AddBlock(block_id="a1", config={"offset": 5})
        cfg = b.get_config()
        assert cfg["block_type"] == "test/add"
        assert cfg["block_id"] == "a1"
        assert cfg["config"] == {"offset": 5}


class TestBlockToDevice:
    def test_to_returns_self(self):
        b = IdentityBlock()
        assert b.to("cpu") is b

    def test_to_propagates_to_sub_blocks(self):
        devices = []
        original_to = IdentityBlock.to
        IdentityBlock.to = lambda self, device: (devices.append(device), self)[1]
        try:
            b = BlockWithSub()
            b.to("cuda")
        finally:
            IdentityBlock.to = original_to


class TestBlockLoadStateStrictEmpty:
    def test_load_empty_state_strict_on_stateless_block(self):
        b = IdentityBlock()
        b.load_state_dict({}, strict=True)

    def test_load_extra_key_strict_on_stateless_block(self):
        b = IdentityBlock()
        with pytest.raises(KeyError):
            b.load_state_dict({"foo": 1}, strict=True)


class TestBlockTrainableParameters:
    def test_default_empty(self):
        b = IdentityBlock()
        assert list(b.trainable_parameters()) == []


class TestBlockFreezeUnfreeze:
    def test_freeze_propagates_to_sub_blocks(self):
        b = BlockWithSub()
        b.freeze()
        assert b.frozen is True
        assert b.child.frozen is True

    def test_unfreeze_propagates_to_sub_blocks(self):
        b = BlockWithSub()
        b.freeze()
        b.unfreeze()
        assert b.frozen is False
        assert b.child.frozen is False


class TestBlockDefaultBlockType:
    def test_default_block_type_is_class_name(self):
        from yggdrasill.foundation.block import AbstractBaseBlock
        class RawBlock(AbstractBaseBlock):
            def forward(self, inputs):
                return inputs
        b = RawBlock()
        assert b.block_type == "RawBlock"
