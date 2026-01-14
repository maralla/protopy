from typing import cast


class _Meta(type):
    symbol_name: str

    def __repr__(cls) -> str:
        return cls.__name__

    def is_terminal(cls) -> bool:
        return issubclass(cls, Terminal)

    def is_nonterminal(cls) -> bool:
        return issubclass(cls, NonTerminal)

    def as_terminal(cls) -> type[Terminal]:
        """Return self as Terminal type, raising TypeError if not one.

        Uses cast() because after the runtime check, cls is guaranteed to be
        a Terminal type, but mypy cannot infer this from the is_terminal() check.
        """
        if not cls.is_terminal():
            msg = f"{cls} is not a Terminal"
            raise TypeError(msg)
        return cast("type[Terminal]", cls)

    def as_nonterminal(cls) -> NonTerminal:
        """Return self as NonTerminal type, raising TypeError if not one.

        Uses cast() because after the runtime check, cls is guaranteed to be
        a NonTerminal type, but mypy cannot infer this from the is_nonterminal() check.
        """
        msg = f"{cls} is not a NonTerminal"
        raise TypeError(msg)


class Terminal(metaclass=_Meta):
    """Base class for terminal symbols in the grammar.

    Examples:
        class ENUM(Terminal, name="enum"): pass
        class IDENT(Terminal): pass  # name defaults to "IDENT"

    """

    name: str

    def __init_subclass__(cls, name: str | None = None, **kwargs: object) -> None:
        """Automatically set name from class name if not provided."""
        if name is not None:
            cls.name = name
        elif not hasattr(cls, 'name'):
            cls.name = cls.__name__

        cls.symbol_name = cls.name

        super().__init_subclass__(**kwargs)


class NonTerminal:
    """Base class for non-terminal symbols in the grammar."""

    non_terminal_symbol_name: str | None = None

    @classmethod
    def grammar_symbol(cls) -> NonTerminal:
        """Create a grammar symbol instance for this NonTerminal class.

        Returns a lightweight instance used only for grammar structure,
        not for actual parsing. Uses empty/default values for all fields.
        """
        instance = object.__new__(cls)
        instance.non_terminal_symbol_name = cls.__name__
        return instance

    @property
    def symbol_name(self) -> str:
        return self.non_terminal_symbol_name or self.__class__.__name__

    def is_terminal(self) -> bool:
        return False

    def is_nonterminal(self) -> bool:
        return True

    def as_nonterminal(self) -> NonTerminal:
        return self

    def as_terminal(self) -> type[Terminal]:
        raise TypeError(f"{self.__class__} is not Terminal")

    def __eq__(self, other: object) -> bool:
        """Compare NonTerminals by symbol_name, not by field values.

        This allows grammar_symbol() instances to match even though they
        are different object instances.
        """
        if not isinstance(other, NonTerminal):
            return NotImplemented
        return self.symbol_name == other.symbol_name

    def __hash__(self) -> int:
        """Hash by symbol_name for use in sets and dicts."""
        return hash(self.symbol_name)


# Type alias for symbols: Terminal types (classes) and NonTerminal instances
Symbol = type[Terminal] | NonTerminal
