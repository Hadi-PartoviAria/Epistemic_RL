use proc_macro::TokenStream;
use quote::quote;

#[proc_macro_derive(Language)]
pub fn language_derive(input: TokenStream) -> TokenStream {
    let ast = syn::parse(input).unwrap();
    impl_language(&ast)
}

fn impl_language(ast: &syn::DeriveInput) -> TokenStream {
    let name = &ast.ident;
    let gen = quote! {
        impl std::ops::Deref for #name {
            type Target = Base;
            fn deref(&self) -> &Self::Target {
                self.base.deref()
            }
        }
        impl From<#name> for std::rc::Rc<Base> {
            fn from(value: #name) -> Self {
                value.base.clone()
            }
        }
        impl TryFrom<std::rc::Rc<Base>> for #name {
            type Error = &'static str;

            fn try_from(value: std::rc::Rc<Base>) -> Result<Self, Self::Error> {
                if Self::check(&value) {
                    Ok( #name { base: value.clone() })
                } else {
                    Err("Malformed formula.")
                }
            }
        }
        impl TryFrom<Box<Base>> for #name {
            type Error = &'static str;

            fn try_from(value: Box<Base>) -> Result<Self, Self::Error> {
                let rc = std::rc::Rc::new(*value);
                rc.try_into()
            }
        }
        impl TryFrom<Base> for #name {
            type Error = &'static str;

            fn try_from(value: Base) -> Result<Self, Self::Error> {
                let rc = std::rc::Rc::new(value);
                rc.try_into()
            }
        }
        impl std::fmt::Display for #name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                let x: std::rc::Rc<Base> = self.to_owned().into();
                x.fmt(f)
            }
        }
        impl std::str::FromStr for #name {
            type Err = &'static str;

            fn from_str(s: &str) -> Result<Self, Self::Err> {
                let base = std::rc::Rc::new(Base::from_str(s)?);
                base.try_into()
            }
        }
        impl serde::Serialize for #name {
            fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
            where
                S: serde::Serializer,
            {
                serializer.serialize_str(&self.to_string())
            }
        }
        impl<'de> serde::Deserialize<'de> for #name {
            fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                deserializer.deserialize_str(crate::utils::parsing::LangVisitor::default())
            }
        }
    };
    gen.into()
}
