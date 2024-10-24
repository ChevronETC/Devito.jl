import Base.==

export FloatX8, FloatX16

struct FloatX{m, M, T, U} <: Real
    value::U
end

FloatX{m, M, T, U}(x::T) where {m, M, T<:Real, U} = FloatX{m, M, T, U}(compress(FloatX{m, M, T, U}, x))

FloatX16(m::T, M::T) where T = FloatX{m, M, T, UInt16}
FloatX8(m::T, M::T) where T = FloatX{m, M, T, UInt8}

# Type based
nbytes(::Type{FloatX{m, M, T, U}}) where {m, M, T<:Real, U} = sizeof(U) * 8
offset(::Type{FloatX{m, M, T, U}}) where {m, M, T<:Real, U} = m
scale(::Type{FloatX{m, M, T, U}}) where {m, M, T<:Real, U} = (M - m) / (2^nbytes(FloatX{m, M, T, U}) - 1)
compressed_type(::Type{FloatX{m, M, T, U}}) where {m, M, T<:Real, U} = U

# Value based
nbytes(x::FloatX{m, M, T, U}) where {m, M, T<:Real, U} = sizeof(U) * 8
offset(x::FloatX{m, M, T, U}) where {m, M, T<:Real, U} = m
scale(x::FloatX{m, M, T, U}) where {m, M, T<:Real, U} = (M - m) / (2^nbytes(x) - 1)
compressed_type(x::FloatX{m, M, T, U}) where {m, M, T<:Real, U} = U

# Conversions
decompress(x::FloatX{m, M, T, U}) where {m, M, T<:Real, U} = m + x.value * scale(x)
compress(::Type{FloatX{m, M, T, U}}, x::T) where {m, M, T<:Real, U} = trunc(U, (x - m) / scale(FloatX{m, M, T, U}))

Base.convert(::Type{FloatX{m, M, T, U}}, x::T) where {m, M, T<:Real, U} = FloatX{m, M, T, U}(x)
Base.convert(::Type{T}, x::FloatX{m, M, T, U}) where {m, M, T<:Real, U} = Base.convert(T, decompress(x))

# Addition
Base.:+(x::FloatX{m, M, T, U}, y::FloatX{m, M, T, U}) where {m, M, T<:Real, U} = compress(FloatX{m, M, T, U}, decompress(x) + decompress(y))
Base.:+(x::FloatX{m, M, T, U}, y::T) where {m, M, T<:Real, U} = compress(FloatX{m, M, T, U}, decompress(x) + y)
Base.:+(x::T, y::FloatX{m, M, T, U}) where {m, M, T<:Real, U} = compress(FloatX{m, M, T, U}, x + decompress(y))

# Subtraction
Base.:-(x::FloatX{m, M, T, U}, y::FloatX{m, M, T, U}) where {m, M, T<:Real, U} = compress(FloatX{m, M, T, U}, decompress(x) - decompress(y))
Base.:-(x::FloatX{m, M, T, U}, y::T) where {m, M, T<:Real, U} = compress(FloatX{m, M, T, U}, decompress(x) - y)
Base.:-(x::T, y::FloatX{m, M, T, U}) where {m, M, T<:Real, U} = compress(FloatX{m, M, T, U}, x - decompress(y))

# Multiplication
Base.:*(x::FloatX{m, M, T, U}, y::FloatX{m, M, T, U}) where {m, M, T<:Real, U} = compress(FloatX{m, M, T, U}, decompress(x) * decompress(y))
Base.:*(x::FloatX{m, M, T, U}, y::T) where {m, M, T<:Real, U} = compress(FloatX{m, M, T, U}, decompress(x) * y)
Base.:*(x::T, y::FloatX{m, M, T, U}) where {m, M, T<:Real, U} = compress(FloatX{m, M, T, U}, x * decompress(y))

# Division
Base.:/(x::FloatX{m, M, T, U}, y::FloatX{m, M, T, U}) where {m, M, T<:Real, U} = compress(FloatX{m, M, T, U}, decompress(x) / decompress(y))
Base.:/(x::FloatX{m, M, T, U}, y::T) where {m, M, T<:Real, U} = compress(FloatX{m, M, T, U}, decompress(x) / y)
Base.:/(x::T, y::FloatX{m, M, T, U}) where {m, M, T<:Real, U} = compress(FloatX{m, M, T, U}, x / decompress(y))

# Define the zero value for FloaXt8
Base.zero(::Type{FloatX{m, M, T, U}}) where {m, M, T<:Real, U} = FloatX{m, M, T, U}(U(0))

# Define the one value for FloaXt8
Base.one(::Type{FloatX{m, M, T, U}}) where {m, M, T<:Real, U} = FloatX{m, M, T, U}(U(1))

# Promotion
Base.promote_rule(::Type{FloatX{m, M, T, U}}, ::Type{S}) where {m, M, T<:Real, U, S<:Real} = FloatX{m, M, promote_type(T,S), U}
Base.promote_rule(::Type{FloatX{m, M, T, U}}, ::Type{FloatX{m, M, S, U}}) where {m, M, T<:Real, U, S<:Real} = FloatX{m, M, promote_type(T, S), U}
Base.promote_rule(::Type{FloatX{m, M, T, U}}, ::Type{FloatX{m, M, T, V}}) where {m, M, T<:Real, U, V} = FloatX{m, M, T, promote_type(U, V)}
Base.promote_rule(::Type{FloatX{m, M, T, U}}, ::Type{FloatX{m, M, S, V}}) where {m, M, T<:Real, U, S<:Real, V} = FloatX{m, M, promote_type(T, S), promote_type(U, V)}

# Comparisson
==(x::FloatX{m, M, T, U}, y::FloatX{m, M, T, U}) where {m, M, T<:Real, U} = decompress(x) == decompress(y)
==(x::FloatX{m, M, T, U}, y::T) where {m, M, T<:Real, U} = decompress(x) == y
==(x::T, y::FloatX{m, M, T, U}) where {m, M, T<:Real, U} = x == decompress(y)

Base.isapprox(x::FloatX{m, M, T, U}, y::FloatX{m, M, T, U}; kwargs...) where {m, M, T<:Real, U} = isapprox(decompress(x), decompress(y); kwargs...)
Base.isapprox(x::FloatX{m, M, T, U}, y::T; kwargs...) where {m, M, T<:Real, U} = isapprox(decompress(x), y; kwargs...)
Base.isapprox(x::T, y::FloatX{m, M, T, U}; kwargs...) where {m, M, T<:Real, U} = isapprox(x, decompress(y); kwargs...)