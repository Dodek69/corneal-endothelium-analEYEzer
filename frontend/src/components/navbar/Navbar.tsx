import Link from 'next/link';
import React from 'react';
import Image from 'next/image';
import styles from './Navbar.module.css';

interface NavItemProps {
    href: string;
    title: string;
}

const NavItem: React.FC<NavItemProps> = ({ href, title }) => (
    <Link href={href} className="text-gray-300 hover:text-white px-3 py-2 rounded-md text-sm font-medium">
        {title}
    </Link>
);

const Navbar: React.FC = () => {
    return (
        <nav className="bg-[#051528] shadow-lg">
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                <div className="flex items-center justify-between h-16">
                    <div className="flex items-center">
                        <div className="flex-shrink-0">
                            <Link href="/">
                                <Image
                                    src="/logo.png" // Replace with your logo's path
                                    alt="Logo"
                                    width={50} // Set the width of the logo
                                    height={50} // Set the height of the logo
                                />
                            </Link>
                        </div>
                        <div className="hidden md:block">
                            <div className="ml-10 flex items-baseline space-x-4">
                                <NavItem href="/" title="Home" />
                                <NavItem href="/about" title="About" />
                                <NavItem href="/services" title="Services" />
                                <NavItem href="/contact" title="Contact" />
                                {/* Add more NavItems as needed */}
                            </div>
                        </div>
                    </div>
                    {/* Optional: if you have a mobile menu, place toggle here */}
                </div>
            </div>
        </nav>
    );
};

export default Navbar;
