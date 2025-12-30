import os

import requests
from fastapi import Header, HTTPException
from jose import jwt

AUTH0_ISSUER = f"https://{os.getenv('AUTH0_DOMAIN')}/"
AUTH0_AUDIENCE = os.getenv("AUTH0_AUDIENCE")
CLAIMS_NAMESPACE = os.getenv("AUTH0_CLAIMS_NAMESPACE")
ALGORITHMS = ["RS256"]

# Cache for JWKS keys to avoid hitting Auth0 on every request
jwks_cache = {}

def get_jwks():
    """Fetches the JSON Web Key Set from Auth0."""
    global jwks_cache
    if not jwks_cache:
        try:
            url = f"{AUTH0_ISSUER}/.well-known/jwks.json"
            jwks_cache = requests.get(url).json()
        except Exception as e:
            print(f"Error fetching JWKS: {e}")
            raise HTTPException(status_code=500, detail="Could not verify authentication keys") from e
    return jwks_cache

def get_current_user(authorization: str | None = Header(None)) -> tuple[str, str]:
    """
    Validates the Bearer token signature using Auth0 JWKS and returns (user_hash, email).
    """
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header missing")

    try:
        token = authorization.split(" ")[1]

        # find the matching key in Auth0's JWKS
        jwks = get_jwks()

        # get the Key ID (kid) from the unverified header
        unverified_header = jwt.get_unverified_header(token)

        rsa_key = {}
        for key in jwks["keys"]:
            if key["kid"] == unverified_header["kid"]:
                rsa_key = {
                    "kty": key["kty"],
                    "kid": key["kid"],
                    "use": key["use"],
                    "n": key["n"],
                    "e": key["e"]
                }
                break

        if not rsa_key:
            # force refresh of cache if key not found (in case of key rotation)
            global jwks_cache
            jwks_cache = {} 
            raise HTTPException(status_code=401, detail="Unable to find appropriate key")

        # cerify the token signature, audience, and issuer
        payload = jwt.decode(
            token,
            rsa_key,
            algorithms=ALGORITHMS,
            audience=AUTH0_AUDIENCE,
            issuer=AUTH0_ISSUER,
        )

        # extract user_id
        user_id = payload.get("sub")
        if not user_id:
            raise HTTPException(status_code=401, detail="Token missing 'sub' claim")
        user_id = user_id.replace("|", "-")

        return user_id

    except jwt.ExpiredSignatureError as e:
        raise HTTPException(status_code=401, detail="Token is expired") from e
    except jwt.JWTClaimsError as e:
        raise HTTPException(status_code=401, detail="Incorrect claims (audience/issuer)") from e
    except Exception as e:
        print(f"Auth Error: {e}")
        raise HTTPException(status_code=401, detail="Could not validate credentials") from e
